# -*- coding: utf-8 -*-
import math 
import unittest
import yaml
from pathlib import Path
from typing import Dict, List, Union, Optional
from typing import List


class ConfigYAML:
    """读取并提供 YAML 配置文件中的值"""

    def __init__(self):
        """初始化：加载 YAML 配置文件"""
        self.config_path = Path(__file__).parent / "reward_conf.yaml"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """私有方法：加载并验证配置文件"""
        # 验证文件存在
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        # 加载并解析 YAML
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 验证解析结果
        if config is None:
            raise ValueError("配置文件解析失败（空文件或格式错误）")
        
        return config

    def get_hard_conf(self) -> Optional[Dict]:
        """获取 hard_conf 节点配置"""
        return self.config.get("hard_conf")

    def get_soft_conf(self) -> Optional[Dict]:
        """获取 soft_conf 节点配置"""
        return self.config.get("soft_conf")

    def get_global_config(self) -> Optional[Dict]:
        """获取 global_config 节点配置"""
        return self.config.get("global_config")

    def get_hard_k_value(self, key: str) -> Optional[int]:
        """获取 hard_conf 中 K 系列（K1/K2/k3）的值"""
        hard_conf = self.get_hard_conf()
        return hard_conf.get(key) if (hard_conf and key in ["K1", "K2", "k3"]) else None

    def get_hard_t_value(self, key: str) -> Optional[float]:
        """获取 hard_conf 中 t 系列（t1/t2/t3）的值"""
        hard_conf = self.get_hard_conf()
        return hard_conf.get(key) if (hard_conf and key in ["t1", "t2", "t3"]) else None

    def get_weights(self) -> Optional[List[List[Union[str, int, float]]]]:
        """获取 hard_conf 中的 weights 二维列表"""
        hard_conf = self.get_hard_conf()
        return hard_conf.get("weights") if hard_conf else None

    def get_early_escalator(self) -> Optional[List[List[Union[str, int, float]]]]:
        """获取 soft_conf 中的 early_escalator 列表"""
        soft_conf = self.get_soft_conf()
        return soft_conf.get("early_escalator") if soft_conf else None

    def get_global_param(self, key: str) -> Optional[Union[int, float]]:
        """获取 global_config 中的参数（step/step_T/step_Tw）"""
        global_cfg = self.get_global_config()
        return global_cfg.get(key) if (global_cfg and key in ["step", "step_T", "step_Tw"]) else None

class GlobalConfig:
    def __init__(self,all_step ):
        self.global_all_step  = all_step
        self.global_config = ConfigYAML()
        self.current_step = 0 
    def getConfig(self)->ConfigYAML:
        return self.global_config
    
    
    ##整体权重
    def getWeights(self, current_step) -> List:
        if current_step < 0 or current_step > self.global_all_step:
            error_msg = (f"输入的current_step无效: {current_step}。"
                        f"有效范围应为0到{self.global_all_step}（包含两端）")
            print(error_msg)
            raise ValueError(error_msg)

        weights = [-1.0]
        # 获取配置实例（简化后续调用）
        config = self.getConfig()
        # 获取全局配置字典
        global_config = config.get_global_config()
        # 获取权重列表
        weights_list = config.get_weights()
        # 获取early_escalator列表
        escalator_list = config.get_early_escalator()

        # 计算区间阈值
        step_T_threshold = global_config.get("step_T", 0) * self.global_all_step
        step_Fromat_threshold = global_config.get("step_Fromat", 0) * self.global_all_step
        step_Tw_threshold = global_config.get("step_Tw", 0) * self.global_all_step

        # 四个区间判断（修复方法调用和索引）
        if current_step < step_T_threshold:
            # 注意：原代码取索引[0]是字符串（如"W1"），这里修正为取数值索引[1]
            weights.extend([
                float(weights_list[0][1]),
                float(weights_list[1][1]),
                float(weights_list[2][1]),
                float(weights_list[3][1]),
                float(escalator_list[0][1]),  # 取数值部分（索引1）
                float(escalator_list[1][1])
            ])
        elif current_step < step_Fromat_threshold:
            weights.extend([
                float(weights_list[0][2]),
                float(weights_list[1][2]),
                float(weights_list[2][2]),
                float(weights_list[3][2]),
                float(escalator_list[0][2]),
                float(escalator_list[1][2])
            ])
        elif current_step < step_Tw_threshold:
            weights.extend([
                float(weights_list[0][3]),
                float(weights_list[1][3]),
                float(weights_list[2][3]),
                float(weights_list[3][3]),
                float(escalator_list[0][2]),
                float(escalator_list[1][2])
            ])
        else:
            weights.extend([
                float(weights_list[0][4]),
                float(weights_list[1][4]),
                float(weights_list[2][4]),
                float(weights_list[3][4]),
                0.0,
                0.0
            ])

        return weights
   
    ##权重更新时间的时间节点
    def getStep_T(self)->int:
        config = self.getConfig()
        global_config = config.get_global_config()
        step_T_threshold = global_config.get("step_T", 0) * self.global_all_step
        return step_T_threshold 
    def getStep_Fromat(self)->int:
        config = self.getConfig()
        global_config = config.get_global_config()
        step_Fromat_threshold = global_config.get("step_Fromat", 0) * self.global_all_step
        return step_Fromat_threshold
    def getStep_Tw(self)->int:
        config = self.getConfig()
        global_config = config.get_global_config()
        step_Tw_threshold = global_config.get("step_Tw", 0) * self.global_all_step
        return step_Tw_threshold

def test():
    cfg = GlobalConfig(1000)
    config = cfg.getConfig()
    
    # 获取整个节点
    hard_conf = config.get_hard_conf()
    print(f"hard_conf${hard_conf}")
    soft_conf = config.get_soft_conf()
    print(f"soft_conf${soft_conf}")
    global_cfg = config.get_global_config()
    print(f"global_cfg${global_cfg}")
    # 获取单个参数
    k1 = config.get_hard_k_value("K1")
    t2 = config.get_hard_t_value("t2")
    step = config.get_global_param("step")
    
    # 获取列表类参数
    weights = config.get_weights()
    escalator = config.get_early_escalator()


    weight1=cfg.getWeights(20)
    weight2=cfg.getWeights(150)
    weight3=cfg.getWeights(300)
    weight4=cfg.getWeights(600)

    print(f"width1 ${weight1}")
    print(f"width2 ${weight2}")
    print(f"width3 ${weight3}")
    print(f"width4 ${weight4}")

    print(f"T:${cfg.getStep_T()}")
    print(f"T:${cfg.getStep_Fromat()}")
    print(f"T:${cfg.getStep_Tw()}")


if __name__ == "__main__":
    test()