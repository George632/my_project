# import numpy as np

# # 已知信息
# center_point = np.array([1.0, 0.0, 0.0])
# direction_vector = np.array([1, 0.5, 0])  # 方向向量可以是单位向量或非单位向量

# segment_length = 2.0

# # 计算半段长度
# half_segment_length = segment_length / 2

# # 计算线段的两个端点
# endpoint1 = center_point - half_segment_length * direction_vector
# endpoint2 = center_point + half_segment_length * direction_vector

# print("Endpoint 1:", endpoint1)
# print("Endpoint 2:", endpoint2)
# distance = np.linalg.norm(endpoint2 - endpoint1)
# print(distance)


import json, matplotlib.pyplot as plt

data = json.load(open("C:/Users/DELL/Desktop/trainer_state.json"))
steps, loss = [], []
for x in data["log_history"]:
    if "loss" in x:
        steps.append(x["step"])
        loss.append(x["loss"])

plt.plot(steps, loss)
plt.xlabel("step")
plt.ylabel("DPO loss")
plt.savefig("loss_curve.png")