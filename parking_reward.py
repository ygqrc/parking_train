# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import math
import numpy as np
import yaml

# ---------- 小工具 ----------
def clip01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)



def stable_sigmoid(z: float) -> float:
    # 数值稳定的 sigmoid
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    else:
        ez = np.exp(z)
        return float(ez / (1.0 + ez))
    
def sigmoid(a ,t) ->float:
    return stable_sigmoid(a*t)
# ---------- 配置 ----------
class ConfigYAML:
    """读取并提供 YAML 配置文件中的值"""
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.config_path = Path(path) if path else (Path(__file__).parent / "reward_conf.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError("配置文件解析失败（空文件或格式错误）")
        return cfg

    def get_hard_conf(self) -> Optional[Dict]:
        return self.config.get("hard_conf")

    def get_soft_conf(self) -> Optional[Dict]:
        return self.config.get("soft_conf")

    def get_global_config(self) -> Optional[Dict]:
        return self.config.get("global_config")

    def get_hard_k_value(self, key: str) -> Optional[Union[int, float]]:
        hard = self.get_hard_conf() or {}
        # 支持 K1/K2/k3
        return hard.get(key)

    def get_hard_t_value(self, key: str) -> Optional[float]:
        hard = self.get_hard_conf() or {}
        return hard.get(key)

    def get_weights(self) -> List[List[Union[str, float]]]:
        hard = self.get_hard_conf() or {}
        ws = hard.get("weights")
        if not ws or len(ws) < 4:
            raise ValueError("hard_conf.weights 配置不完整，需要至少 4 行（W1~W4）")
        return ws

    def get_early_escalator(self) -> Dict[str, float]:
        # 实际是 dict（W5/W6），修正文档与注解
        soft = self.get_soft_conf() or {}
        esc = soft.get("early_escalator") or {}
        # 兜底为 0.0，避免 KeyError
        return {"W5": float(esc.get("W5", 0.0)),
                "W6": float(esc.get("W6", 0.0))}

    def get_global_param(self, key: str) -> Optional[Union[int, float]]:
        g = self.get_global_config() or {}
        return g.get(key)
    def _get_hard_reward(self) -> Dict:
        """获取 hard_reward 根节点（内部辅助方法）"""
        return self.config.get("hard_reward", {})

    def get_r1_soft_open(self) -> float:
        """获取 R1_SOFT 中的 R_P_OPEN 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R1", {}).get("R1_SOFT", {}).get("R_P_OPEN", 0.0))

    def get_r1_soft_close(self) -> float:
        """获取 R1_SOFT 中的 R_P_CLOSE 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R1", {}).get("R1_SOFT", {}).get("R_P_CLOSE", 0.0))

    def get_r1_soft_size(self) -> float:
        """获取 R1_SOFT 中的 R_P_SIZE 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R1", {}).get("R1_SOFT", {}).get("R_P_SIZE", 0.0))

    def get_r1_hard(self) -> float:
        """获取 R1_HARD 奖励值，默认 -1.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R1", {}).get("R1_HARD", -1.0))

    def get_r2_parser(self) -> float:
        """获取 R2 中的 R2_PARSER 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R2", {}).get("R2_PARSER", 0.0))

    def get_r2_p_len(self) -> float:
        """获取 R2 中的 R2_P_LEN 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R2", {}).get("R2_P_LEN", 0.0))

    def get_r2_p_unique(self) -> float:
        """获取 R2 中的 R2_P_UNIQUE 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R2", {}).get("R2_P_UNIQUE", 0.0))

    def get_r3_lie(self) -> float:
        """获取 R3 中的 R3_LIE 惩罚值，默认 -1.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R3", {}).get("R3_LIE", -1.0))

    def get_r3_prec(self) -> float:
        """获取 R3 中的 R3_PREC 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R3", {}).get("R3_PREC", 0.0))

    def get_r3_jacrard(self) -> float:
        """获取 R3 中的 R3_JACRARD 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R3", {}).get("R3_JACRARD", 0.0))

    def get_r4_s(self) -> float:
        """获取 R4 中的 R4_S 奖励值，默认 0.0"""
        hard_reward = self._get_hard_reward()
        return float(hard_reward.get("R4", {}).get("R4_S", 0.0))




# ---------- 全局逻辑 ----------
class GlobalConfig:
    def __init__(self, all_step: int, config: Optional[ConfigYAML] = None):
        if all_step <= 0:
            raise ValueError("all_step 必须为正整数")
        self.global_all_step = int(all_step)
        self._config = config or ConfigYAML()

    def getConfig(self) -> ConfigYAML:
        return self._config

    # 阶段化权重（返回正好 4 个数）
    def getWeights(self, current_step: int) -> List[float]:
        if current_step < 0 or current_step > self.global_all_step:
            raise ValueError(f"current_step 超界：{current_step}，应在 [0, {self.global_all_step}]")

        cfg = self.getConfig()
        g = cfg.get_global_config() or {}
        ws = cfg.get_weights()

        step_T_th      = float(g.get("step_T", 0.0))      * self.global_all_step
        step_Fromat_th = float(g.get("step_Fromat", 0.0)) * self.global_all_step  # 与 YAML 保持拼写一致
        step_Tw_th     = float(g.get("step_Tw", 0.0))     * self.global_all_step

        # ws[i] 结构类似: [ "W1", v_T, v_Fromat, v_Tw, v_final ]
        if current_step < step_T_th:
            col = 1
        elif current_step < step_Fromat_th:
            col = 2
        elif current_step < step_Tw_th:
            col = 3
        else:
            col = 4

        return [float(ws[0][col]), float(ws[1][col]), float(ws[2][col]), float(ws[3][col])]

    # 各阶段阈值（绝对步数）
    def getStep_T(self) -> int:
        g = self.getConfig().get_global_config() or {}
        return int(float(g.get("step_T", 0.0)) * self.global_all_step)

    def getStep_Fromat(self) -> int:
        g = self.getConfig().get_global_config() or {}
        return int(float(g.get("step_Fromat", 0.0)) * self.global_all_step)

    def getStep_Tw(self) -> int:
        g = self.getConfig().get_global_config() or {}
        return int(float(g.get("step_Tw", 0.0)) * self.global_all_step)

    # a(t), tau(t)
    def get_soft_tau(self, step: int) -> float:
        soft = self.getConfig().get_soft_conf() or {}
        T0 = float(soft.get("T0", 0.0)); T1 = float(soft.get("T1", 0.0))
        return float(T0 + (T1 - T0) * step / self.global_all_step)

    def get_soft_a(self, step: int) -> float:
        soft = self.getConfig().get_soft_conf() or {}
        A0 = float(soft.get("A0", 1.0)); A1 = float(soft.get("A1", 1.0))
        return float(A0 + (A1 - A0) * step / self.global_all_step)

    # gate(t)
    def get_soft_gate(self, R2_SOFT: float, R3_SOFT: float, step: int) -> float:
        soft = self.getConfig().get_soft_conf() or {}
        alpha = float(soft.get("R2VR3", 0.5))
        r2 = clip01(R2_SOFT); r3 = clip01(R3_SOFT)
        smix = alpha * r2 + (1 - alpha) * r3
        z = self.get_soft_a(step) * (smix - self.get_soft_tau(step))
        return stable_sigmoid(z)

    # 早期扶梯 + 余弦退火
    def _anneal(self, base: float, step: int) -> float:
        g = self.getConfig().get_global_config() or {}
        T_end = float(g.get("step_Tw", 0.0)) * self.global_all_step
        if T_end <= 0:
            # 没有设置退火窗口就直接返回 base
            return float(base)
        C = min(1.0, max(0.0, step / T_end))
        return float(0.5 * (1.0 + math.cos(math.pi * C)) * base)

    def get_w5_SA(self, step: int) -> float:
        esc = self.getConfig().get_early_escalator()
        return self._anneal(esc["W5"], step)

    def get_w6_SA(self, step: int) -> float:
        esc = self.getConfig().get_early_escalator()
        return self._anneal(esc["W6"], step)

    # Aux(t) = gate(t) * ( w5(t)*R2 + w6(t)*R3 )
    def aux_all(self, R2_SOFT: float, R3_SOFT: float, step: int) -> float:
        w5 = self.get_w5_SA(step); w6 = self.get_w6_SA(step)
        gate = self.get_soft_gate(R2_SOFT, R3_SOFT, step)
        return float(gate * (w5 * R2_SOFT + w6 * R3_SOFT))




if __name__ == "__main__":
    cfg = GlobalConfig(1000)
    config = cfg .getConfig()
    print(config.get_r1_soft_open())    # 输出 0.4
    print(config.get_r1_soft_close())   # 输出 0.4
    print(config.get_r1_soft_size())    # 输出 0.2

    # 获取 R1 硬奖励值
    print(config.get_r1_hard())         # 输出 -1.0

    # 获取 R2~R4 相关值
    print(config.get_r2_parser())       # 输出 1.0
    print(config.get_r2_p_len())       # 输出 1.0
    print(config.get_r2_p_unique())          # 输出 -1.0


    print(config.get_r3_lie())       # 输出 1.0
    print(config.get_r3_prec())       # 输出 1.0
    print(config.get_r3_jacrard())          # 输出 -1.0



    print(config.get_r4_s())            # 输出 2.0

    print("阶段阈值（绝对步）:",
          cfg.getStep_T(), cfg.getStep_Fromat(), cfg.getStep_Tw())

    for step in [20, 150, 300, 600]:
        print(f"step={step} -> W1~W4:", cfg.getWeights(step))

    for (step, r2, r3) in [(10, 0.8, 0.5), (10, 0.0, 0.5), (10, 0.8, 0.0),
                           (120, 0.8, 0.5), (240, 0.0, 0.0), (600, 0.8, 0.5)]:
        score = cfg.aux_all(r2, r3, step)
        print(f"在第 {step} 步：R2={r2}, R3={r3} -> Aux={score:.6f}, "
              f"gate={cfg.get_soft_gate(r2,r3,step):.6f}, "
              f"w5={cfg.get_w5_SA(step):.6f}, w6={cfg.get_w6_SA(step):.6f}")
