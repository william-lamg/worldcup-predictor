# worldcup-predictor 鈥?World Cup 2026 AI Match Prediction Engine

**Elo + XGBoost + Market Odds Ensemble**, 寰炲皬绲勮辰鍒版窐姹拌辰锛屾柟鍚戞纰虹巼 **73.3%**锛?0 鍫村闅涙瘮璩介璀夛級銆?
> **鏈€鏂扮増鏈細v6.2** 鈥?姹鸿辰鏁欒〒淇锛堝穿娼版晥鎳夈€侀€ｇ簩鍔犳檪鎳茬桨銆佸毚鍘茶鍒ゃ€佹焙璩?xG cap锛?> 瑭宠 [CHANGELOG.md](CHANGELOG.md)

---

## 馃彈锔?鏋舵

```
                      鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                      鈹?   Data Layer      鈹?                      鈹? The Odds API      鈹?                      鈹? 44k Dataset       鈹?                      鈹? 绔跺僵 SP (婢冲)     鈹?                      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                               鈹?                      鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                      鈹?  Analysis Core    鈹?                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?     鈹?                      鈹? 鈹?XGBoost  鈹?15%  鈹?鈫?KO 妯″紡
                      鈹? 鈹?Elo V3   鈹?20%  鈹?                      鈹? 鈹?Market   鈹?65%  鈹?                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?     鈹?                      鈹?鈹屸攢鈹€ v6.0 H2H 鈹€鈹€鈹€鈹€鈹?鈹?                      鈹?鈹?  44k 寰€绺惧堡   鈹?鈹?                      鈹?鈹溾攢鈹€ v6.1 Clutch 鈹€鈹?鈹?                      鈹?鈹?  绲曟鍩哄洜     鈹?鈹?                      鈹?鈹斺攢鈹€ v6.2 淇 鈹€鈹€鈹€鈹?鈹?                      鈹?   宕╂桨鏁堟噳       鈹?                      鈹?   鍔犳檪鎳茬桨       鈹?                      鈹?   鍤村幉瑁佸垽       鈹?                      鈹?   姹鸿辰 xG cap    鈹?                      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                               鈹?                      鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                      鈹?  Output Layer     鈹?                      鈹? HTML Reports      鈹?                      鈹? 绔跺僵鎺ㄨ枽鏍煎紡       鈹?                      鈹? TEMP-QUIZ 娆勪綅    鈹?                      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

---

## 馃攽 API 瑷疆

鏈爡鐩渶瑕佷互涓嬪厤璨?API Key 鍏堝彲浠ユ甯搁亱琛岋細

| API | 鐢ㄩ€?| 鐢宠珛閫ｇ祼 |
|:----|:----|:-------|
| **The Odds API** | 鍗虫檪甯傚牬璩犵巼 | https://the-odds-api.com/#get-access |

灏?Key 濉叆 `.env` 妾旀锛堝彲瑜囪＝ `.env.example`锛夛細

```bash
cp .env.example .env
# 鐒跺緦绶ㄨ集 .env 濉叆浣犵殑 API Key
```

---

## 馃殌 蹇€熼枊濮?
```bash
pip install -r requirements.txt
cd src
python predictor.py             # 闋愯ō demo 妯″紡
```

### 闋愭脯鍠牬姣旇辰

```python
from predictor import predict_ensemble, EloRatingSystem
import joblib

elo = joblib.load("models/elo_v3.joblib")
model = joblib.load("models/xgb_weighted.joblib")

result = predict_ensemble(
    "Spain", "Argentina",
    elo, model, feat_df,
    market_odds={"h2h_home": 2.35, "h2h_draw": 3.20, "h2h_away": 3.10},
    is_knockout=True,
    # v6.2 鍙冩暩
    away_et_fatigue=3,      # 闃挎牴寤烽€ｇ簩 3 鍫?KO 鍔犳檪
    away_health=2,          # 鐤插嫗
    referee="Vin膷i膰",       # 鍤村幉瑁佸垽
    tournament_stage="final",
)

print(f"xG: {result['xg_home']} vs {result['xg_away']}")
print(f"鏅夌礆: H {result['ko']['advance_home']} A {result['ko']['advance_away']}")
print(f"闋愭湡榛冪墝: {result['exp_yellow']}")
print(f"宕╂桨棰ㄩ毆: {result['v6_2']['collapse_risk_away']}")
```

---

## 馃搳 妯″瀷婕旈€?
| 鐗堟湰 | 鏃ユ湡 | 鏍稿績鏀归€?|
|:-----|:-----|:---------|
| **v6.2** | 2026-07-20 | 姹鸿辰鏁欒〒淇锛氬穿娼版晥鎳夈€侀€ｇ簩鍔犳檪鎳茬桨銆佸毚鍘茶鍒ゃ€佹焙璩?xG cap |
| **v6.1** | 2026-07-16 | Clutch gene锛堢禃娈哄熀鍥狅級銆乺ecency H2H |
| **v6.0** | 2026-07-15 | 灏嶈辰姝峰彶灞わ紙44k dataset锛夈€丠2H override |
| **v5.4** | 2026-07-12 | Health factor銆丒lo 娈樺樊鏍℃銆佹法鍕濈悆鍋忕Щ |
| **v5.0** | 2026-07-06 | 瀹氭€у垎鏋愬堡锛堝偡鐥呫€佸搧鐗屾孩鍍广€侀槻瀹堛€佹捣鎷斻€佹埌鎰忥級 |
| **v4.0** | 2026-06-15 | Ensemble 鏋舵銆丏raw calibration |
| **v3.0** | 2026-06-14 | XGBoost 妯″瀷瑷撶反 |

---

## 馃И 椹楄瓑绲愭灉

### 2026 涓栫晫鐩冨鎴拌〃鐝?
| 闅庢 | 鍫存 | 鍛戒腑 | 鍛戒腑鐜?|
|:-----|:----:|:----:|:------:|
| 灏忕祫璩?| 19 | 15 | 78.9% |
| 娣樻卑璩?| 11 | 7 | 63.6% |
| **绺借▓** | **30** | **22** | **73.3%** |

### v6.2 姹鸿辰閲嶆脯锛堣タ鐝墮 vs 闃挎牴寤凤級

| 鎸囨 | v6.1 | v6.2 | 瀵﹂殯 |
|:-----|:----:|:----:|:----:|
| 闃挎牴寤?xG | 2.01 | **0.73** | 0 灏勯杸 鉁?|
| 鍏堥枊绱€閷?| 闃挎牴寤?| **瑗跨彮鐗?* | 瑗跨彮鐗?鉁?|
| 闋愭湡榛冪墝 | 3.16 | **5.61** | 7+ 鉁?|
| 绱呯墝姒傜巼 | 0.177 | **0.443** | 鎭╀綈绱呯墝 鉁?|

---

## 馃搧 灏堟绲愭

```
worldcup-predictor/
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ predictor.py        鈫?涓绘ā鍨嬶紙v6.2锛?鈹?  鈹溾攢鈹€ daily_predict.py    鈫?姣忔棩闋愭脯娴佹按绶?鈹?  鈹斺攢鈹€ rebuild_elo.py      鈫?Elo 閲嶅缓鑵虫湰
鈹溾攢鈹€ models/
鈹?  鈹溾攢鈹€ elo_v3.joblib       鈫?Elo 瑭曞垎绯荤当
鈹?  鈹斺攢鈹€ xgb_weighted.joblib 鈫?XGBoost 妯″瀷
鈹溾攢鈹€ data/
鈹?  鈹斺攢鈹€ (璩囨枡闆嗭紝闇€鑷涓嬭級)
鈹溾攢鈹€ docs/                   鈫?鏂囨獢
鈹溾攢鈹€ CHANGELOG.md            鈫?鐗堟湰璁婃洿瑷橀寗
鈹斺攢鈹€ README.md
```

---

## 馃敩 闂滈嵉鍑芥暩

### `predict_ensemble()`

鏍稿績闋愭脯鍑芥暩锛屾暣鍚?XGBoost銆丒lo銆佸競鍫磋碃鐜囦笁灞ゃ€?
**v6.2 鏂板鍙冩暩锛?*
- `referee` 鈥?瑁佸垽鍚嶇ū锛圴in膷i膰 / Barton锛?- `tournament_stage` 鈥?璩戒簨闅庢锛坒inal / semi / quarter锛?- `home_et_fatigue` / `away_et_fatigue` 鈥?閫ｇ簩鍔犳檪鍫存
- `home_health` / `away_health` 鈥?鐞冮殜鍋ュ悍鐙€鎱嬶紙0-4锛?
**杓稿嚭娆勪綅锛?*
- `ensemble_home/draw/away` 鈥?90 鍒嗛悩姒傜巼
- `xg_home/xg_away` 鈥?鏈熸湜鍏ョ悆
- `ko` 鈥?娣樻卑璩介€茬▼锛圗T/PK 姒傜巼锛?- `exp_yellow` / `p_red` 鈥?绱呴粌鐗岄爯娓?- `v6_2` 鈥?v6.2 瑾胯│瀛楁锛堝穿娼伴ⅷ闅€佸姞鏅傛嚥缃般€佽鍒や箻鏁革級

---

## 馃摑 License

MIT License 鈥?瑭宠 [LICENSE](LICENSE)

---

## 馃檹 鑷磋瑵

- **The Odds API** 鈥?甯傚牬璩犵巼鏁告摎
- **AndyLin31/International-Football-Results** 鈥?44k 姝峰彶璩芥灉鏁告摎闆?- **openfootball/worldcup.json** 鈥?涓栫晫鐩冭辰绋嬫暩鎿?
---

**Made with 馃 by william-lamg**
