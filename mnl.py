# mnlの実装
import pandas as pd
import numpy as np 
from scipy.optimize import minimize 

df = pd.read_csv('/Users/takahiromatsunaga/Library/CloudStorage/OneDrive-TheUniversityofTokyo/bin_R_ex/mnl/ensyu.csv', encoding='shift-jis')

def fr(x: np.array) -> float:
    b1, b2, b3, b4, d1, f1 = x # b1~b4: 定数項　d1: 所要時間　f1: 運賃

    train: pd.Series = df["代替手段生成可否train"] * np.exp(d1*df["総所要時間train"]/100 + f1*df["費用train"]/100 + b1) # そもそも代表手段生成可否が0なら選択肢にすら入ってこないということ．ここが1の時交通手段の比較対象として上がってくる．この時初めて効用が比較されることになる
    bus: pd.Series = df["代替手段生成可否bus"] * np.exp(d1*df["総所要時間bus"]/100 + f1*df["費用bus"]/100 + b2)
    car: pd.Series = df["代替手段生成可否car"] * np.exp(d1*df["所要時間car"]/100 + b3)
    bike: pd.Series = df["代替手段生成可否bike"] * np.exp(d1*df["所要時間bike"]/100 + b4)
    walk: pd.Series = df["代替手段生成可否walk"] * np.exp(d1*df["所要時間walk"]/100) # 定数項は選択肢数から1減らす

    deno: pd.Series = car + train + bus + bike + walk # 分母

    Ptrain: pd.Series = df["代替手段生成可否train"] * (train/deno) # train の選択確率
    Pbus: pd.Series = df["代替手段生成可否bus"] * (bus/deno)
    Pcar: pd.Series = df["代替手段生成可否car"] * (car/deno)
    Pbike: pd.Series = df["代替手段生成可否bike"] * (bike/deno)
    Pwalk: pd.Series = df["代替手段生成可否walk"] * (walk/deno)

    Ptrain = Ptrain.where(Ptrain != 0, 1)
    Pbus = Pbus.where(Pbus != 0, 1)
    Pcar = Pcar.where(Pcar != 0, 1)
    Pbike = Pbike.where(Pbike != 0, 1)
    Pwalk = Pwalk.where(Pwalk != 0, 1)

    # 選択結果（今回は0or1．同時推定の場合，ここが0-1の連続値をとる）
    Ctrain: pd.Series = df["代表交通手段"] == "鉄道"
    Cbus: pd.Series = df["代表交通手段"] == "バス"
    Ccar: pd.Series = df["代表交通手段"] == "自動車"
    Cbike: pd.Series = df["代表交通手段"] == "自転車"
    Cwalk: pd.Series = df["代表交通手段"] == "徒歩"

    LL: float = np.sum(Ctrain * np.log(Ptrain) + Cbus * np.log(Pbus) + Ccar * np.log(Pcar) + Cbike * np.log(Pbike) + Cwalk * np.log(Pwalk))

    return LL

# 最大尤度（optimizeしてる）とその時のパラメタの推定値xが出る
# ただしscipyにmaxmizeがないので尤度*-1を最小化minimizeしてる点に注意
def mf(x: np.array) -> float:
    return -fr(x)

# パラメタ初期値（全て0）
x0 = np.zeros(6)

res = minimize(mf, x0, method = "Nelder-Mead")
print(res)

