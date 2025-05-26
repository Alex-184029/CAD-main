import ezdxf
from ezdxf.addons import odafc
from typing import List

def convert_dwg_to_dxf(dwg_path, dxf_path):
    # DWG转换为DXF，需要的add-on:odafc
    # 需要下载安装ODAFILECONVERTER
    # 设置ODAFILECONVERTER程序路径
    # value = "\"D:\ODA\ODAFileConverter\ODAFileConverter.exe\""
    print('step1')
    value = r'E:\School\Grad1\CAD\CAD_ltl\CAD-ltl\ODA\ODAFileConverter\ODAFileConverter.exe'
    ezdxf.options.set("odafc-addon", "win_exec_path", value)  # ezdxf.options.set(section, key, value)
    odafc.win_exec_path = r'E:\School\Grad1\CAD\CAD_ltl\CAD-ltl\ODA\ODAFileConverter\ODAFileConverter.exe'
    print('step2')
    # print(ezdxf.options.get("odafc-addon", "win_exec_path", default=""))
    # ezdxf.options.print()

    # Convert
    if ezdxf.addons.odafc.is_installed():
        print('convert begin')
        odafc.convert(dwg_path, dxf_path, version='R2018')
        print('convert finish')
        return True
    else:
        print('step3')
        print("ODAFC尚未安装")
        return False

def coinChange(coins: List[int], amount: int):     # 零钱问题，递归法
    def dp(n):
        # base case
        if n == 0: return 0
        if n < 0: return -1
        # 求最⼩值，所以初始化为正⽆穷
        res = float('INF')
        for coin in coins:
            subproblem = dp(n - coin)
            # ⼦问题⽆解，跳过
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)
        return res if res != float('INF') else -1
    return dp(amount)

def coinChange2(coins: List[int], amount: int):    # 零钱问题，备忘录法
    memo = dict()
    def dp(n):
        if n in memo: return memo[n]
        if n == 0: return 0
        if n < 0: return -1
        res = float('INF')    # 初始正无穷
        for coin in coins:
            subproblem = dp(n - coin)
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)
        # 记入备忘录
        memo[n] = res if res != float('INF') else -1
        print('memo %d: %s' % (n, memo))
        return memo[n]
    return dp(amount)

def coinChange3(coins: List[int], amount: int):   # 零钱问题，dp table法
    dp = [0] + [amount + 1] * amount
    print('dp0:', dp)
    for i in range(amount + 1):
        for coin in coins:
            if i - coin < 0: continue    # 子问题无解，剪枝跳过
            dp[i] = min(dp[i], 1 + dp[i - coin])
    print('dp1:', dp)
    return -1 if dp[amount] == amount + 1 else dp[amount]

def testWalrus():
    a = 1
    res = 'yes' if (b := a) else 'no'
    print('res:', res)
    print('b:', b)

def doTest():
    dicts = []
    dict = { 'apple': 15, 'banana': 20 }
    dict2 = { 'apple': 15, 'banana': 20 }
    dict3 = { 'apple': 15, 'banana': 20 }
    dicts.append(dict)
    dicts.append(dict2)
    dicts.append(dict3)
    
    for dict in dicts:
        dict['orange'] = 25
    
    print('dicts:', dicts)

if __name__ == '__main__':
    doTest()
