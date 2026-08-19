# Distill 代码示例（对照真实实现）

路径前缀：`lerobot_alohamini/examples/alohamini/fastwam_distill/`

---

## 1. Teacher：冻住 + 推动作 chunk

文件：`official_teacher.py`

```python
# 加载：eval，不训
model = instantiate(cfg.model, ...)
model.load_checkpoint(str(ckpt))
model = model.to(device).eval()   # ← 冻结 teacher

# 推理：no_grad → env-space [H, 7]
def predict_env_action_chunk(teacher, obs, task_description, num_inference_steps=10):
    prompt = DEFAULT_PROMPT.format(task=task_description)
    image, proprio, _ = _obs_to_model_input(obs, ...)

    with torch.no_grad():                          # ← 不反传
        pred = teacher.model.infer_action(
            prompt=prompt,
            input_image=image,
            action_horizon=teacher.action_horizon, # ≈32
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            ...
        )
    action = pred["action"]
    action = _denormalize_action(action, teacher.processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1      # gripper 映射
    action = invert_gripper_action(action)         # 对齐 LIBERO
    return np.asarray(action, dtype=np.float32)    # shape [H, 7]
```

**记住：** teacher 只 `infer_action`，dump 前把动作弄到 **env-space**。

---

## 2. Rollout + Dump：环境里跑，按 stride 存样本

文件：`dump_teacher_env.py`

```python
records = {"images": [], "states": [], "tasks": [], "actions": []}

for ep in range(episodes):
    env, task_description = get_libero_env(...)
    obs = env.set_init_state(init)
    action_queue = []
    step = 0

    while step < max_steps:
        if len(action_queue) == 0:
            # --- Teacher 推理一整段 chunk ---
            chunk = predict_env_action_chunk(teacher, obs, task_description)

            # --- Dump：每隔 stride 存一条 (obs → a_teacher) ---
            if step % stride == 0:
                packed = images_from_obs_for_student(obs, task_description)
                records["images"].append(packed)          # 相机 CHW
                records["states"].append(state_tensor)    # eef+quat+gripper
                records["tasks"].append(task_description)
                records["actions"].append(torch.from_numpy(chunk).float())

            # --- 只执行 chunk 前 replan_steps 步，再重新问 teacher ---
            for a in chunk[:replan_steps]:
                action_queue.append(a)

        action = action_queue.pop(0)
        obs, reward, done, info = env.step(action.tolist())  # ← Rollout
        step += 1
        if done:
            break

# --- 落盘 ---
payload = {
    "format": "env_rollout",
    "actions": torch.stack(records["actions"]),  # [N, H, 7]
    "images": records["images"],
    "states": records["states"],
    "tasks": records["tasks"],
}
torch.save(payload, "teacher_actions.pt")
```

| 概念 | 对应代码 |
|------|----------|
| Rollout | `env.step(action)` |
| Dump | `records[...].append(...)` + `torch.save` |
| Teacher | `predict_env_action_chunk` |
| Replan | `chunk[:replan_steps]` 入队 |

---

## 3. Student Dataset：标签 = teacher 动作

文件：`train_distill.py` → `EnvRolloutDataset`

```python
class EnvRolloutDataset(Dataset):
    def __getitem__(self, i):
        item = {}
        item.update(self.images[i])                         # obs 图像
        if self.states[i] is not None:
            item["observation.state"] = self.states[i]       # obs 状态
        item["action"] = _pad_action(self.actions[i], chunk_size)  # ← a_teacher 当标签
        item["task"] = self.tasks[i]
        return item
```

普通 BC 标签是人类演示；这里标签是 **`a_teacher`** → 这就是 distill。

---

## 4. Distill loss：复用 student.forward

文件：`train_distill.py` 训练循环

```python
policy = make_policy(cfg, ...)          # Student = SmolVLA
policy.train()
optim = torch.optim.AdamW(...)

for batch in loader:
    batch = preprocessor(batch)         # 归一化等
    batch = move_to_gpu(batch)

    optim.zero_grad()
    loss, loss_dict = policy.forward(batch)   # ← 标签已是 a_teacher
    loss.backward()
    optim.step()
```

概念上：

```text
L_distill ≈ || a_student(obs) - a_teacher(obs) ||²
         实现上 = student 原生 training loss（标签换成 teacher）
```

**没有**单独写一个 `DistillLoss` 类；换标签 + `forward` 就是动作蒸馏。

---

## 5. 最小可跑伪代码（整条链路）

```python
# ========== DUMP ==========
teacher = load_official_teacher(...).eval()
buf = []
for ep in range(E):
    obs = env.reset()
    while not done:
        with torch.no_grad():
            a_chunk = teacher.infer_action(obs)      # [H,7] env-space
        if step % stride == 0:
            buf.append({"obs": obs, "action": a_chunk})
        for a in a_chunk[:replan]:
            obs, _, done, _ = env.step(a)
torch.save(buf, "teacher_actions.pt")

# ========== TRAIN ==========
student = SmolVLA.from_pretrained("lerobot/smolvla_base")
for batch in DataLoader(TeacherDump("teacher_actions.pt")):
    # batch["action"] == a_teacher
    loss, _ = student.forward(preprocessor(batch))
    loss.backward(); optim.step()

# ========== EVAL ==========
# student 自己在 LIBERO 闭环 → pc_success
```

---

## 6. Shell 怎么跑

```bash
cd /home/july/lerobot_alohamini

# dump（conda: fastwam_official）
bash examples/alohamini/fastwam_distill/1_dump_env.sh
# → outputs/fastwam_distill/teacher_dump/teacher_actions.pt

# train student（conda: lerobot_alohamini）
bash examples/alohamini/fastwam_distill/2_train.sh
# → .../checkpoints/last/pretrained_model

# eval SR
bash examples/alohamini/fastwam_distill/3_eval_sr.sh
# → pc_success
```

---

## 7. 面试 30 秒对着代码讲

1. **Teacher** `infer_action` + `no_grad` → `a_teacher`  
2. **Rollout** `env.step`；**Dump** 存 `(images, state, task, actions)`  
3. **Student** `EnvRolloutDataset` 把 `actions` 塞进 `item["action"]`  
4. **Loss** `policy.forward(batch)`，标签已是 teacher → 动作蒸馏  
5. **Eval** 闭环 SR，不和 97.1% teacher 比 SOTA  

对应文件：`official_teacher.py` → `dump_teacher_env.py` → `train_distill.py` → `3_eval_sr.sh`
