# -*- coding: utf-8 -*-
import math 
import unittest
import yaml
from pathlib import Path

class TestConfigYAML(unittest.TestCase):
    """测试 YAML 配置文件的结构和内容正确性"""

    @classmethod
    def setUpClass(cls):
        """读取 YAML 配置，只执行一次"""
        config_path = Path(__file__).parent / "reward_conf.yaml"
        # 验证文件存在
        cls.assertTrue(config_path.exists(), f"配置文件 {config_path} 不存在")
        
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)
        cls.assertIsNotNone(cls.config, "配置文件解析失败（可能为空或格式错误）")

    def test_hard_conf_structure(self):
        """测试 hard_conf 节点的结构和键值"""
        hard_conf = self.config.get("hard_conf")
        self.assertIsNotNone(hard_conf, "配置中缺少 hard_conf 节点")

        # 验证 K 系列阈值（整数）
        k_keys = ["K1", "K2", "k3"]
        for key in k_keys:
            self.assertIn(key, hard_conf, f"hard_conf 缺少键 {key}")
            self.assertIsInstance(hard_conf[key], int, f"{key} 应为整数类型")

        # 验证 t 系列阈值（浮点数）
        t_keys = ["t1", "t2", "t3"]
        for key in t_keys:
            self.assertIn(key, hard_conf, f"hard_conf 缺少键 {key}")
            self.assertIsInstance(hard_conf[key], float, f"{key} 应为浮点数类型")

        # 验证 weights 二维列表
        weights = hard_conf.get("weights")
        self.assertIsNotNone(weights, "hard_conf 缺少 weights 列表")
        self.assertIsInstance(weights, list, "weights 应为列表类型")
        self.assertEqual(len(weights), 4, "weights 应包含 4 个子列表（W1-W4）")
        
        # 验证每个子列表的结构（[字符串, 数值, 数值, 数值, 数值]）
        for i, item in enumerate(weights, 1):
            self.assertIsInstance(item, list, f"weights 第 {i} 项不是列表")
            self.assertEqual(len(item), 5, f"weights 第 {i} 项应包含 5 个元素")
            self.assertIsInstance(item[0], str, f"weights 第 {i} 项的指标名称应为字符串")
            for val in item[1:]:
                self.assertIsInstance(val, (int, float), f"weights 第 {i} 项的数值应为数字")

    def test_soft_conf_structure(self):
        """测试 soft_conf 节点的结构和键值"""
        soft_conf = self.config.get("soft_conf")
        self.assertIsNotNone(soft_conf, "配置中缺少 soft_conf 节点")

        # 验证 early_escalator 列表（W5-W6）
        early_escalator = soft_conf.get("early_escalator")
        self.assertIsNotNone(early_escalator, "soft_conf 缺少 early_escalator 列表")
        self.assertIsInstance(early_escalator, list, "early_escalator 应为列表类型")
        self.assertEqual(len(early_escalator), 2, "early_escalator 应包含 2 个子列表（W5-W6）")
        
        # 验证每个子列表的结构（[字符串, 数值, 数值]）
        for i, item in enumerate(early_escalator, 1):
            self.assertIsInstance(item, list, f"early_escalator 第 {i} 项不是列表")
            self.assertEqual(len(item), 3, f"early_escalator 第 {i} 项应包含 3 个元素")
            self.assertIsInstance(item[0], str, f"early_escalator 第 {i} 项的指标名称应为字符串")
            for val in item[1:]:
                self.assertIsInstance(val, (int, float), f"early_escalator 第 {i} 项的数值应为数字")

        # 验证 T/A 系列参数（浮点数或整数）
        ta_keys = ["T0", "T1", "A0", "A1"]
        for key in ta_keys:
            self.assertIn(key, soft_conf, f"soft_conf 缺少键 {key}")
            self.assertIsInstance(soft_conf[key], (int, float), f"{key} 应为数字类型")

    def test_global_config_structure(self):
        """测试 global_config 节点的结构和键值"""
        global_config = self.config.get("global_config")
        self.assertIsNotNone(global_config, "配置中缺少 global_config 节点")

        # 验证 step（整数）
        self.assertIn("step", global_config, "global_config 缺少 step 键")
        self.assertIsInstance(global_config["step"], int, "step 应为整数类型")
        self.assertGreater(global_config["step"], 0, "step 应大于 0")

        # 验证 step_T 和 step_Tw（浮点数）
        step_keys = ["step_T", "step_Tw"]
        for key in step_keys:
            self.assertIn(key, global_config, f"global_config 缺少 {key} 键")
            self.assertIsInstance(global_config[key], float, f"{key} 应为浮点数类型")
            self.assertGreaterEqual(global_config[key], 0, f"{key} 不应为负数")

def test():
    # Test configuration file
    unittest.main(verbosity=2)  # Output detailed test results

if __name__ == "__main__":
    test()