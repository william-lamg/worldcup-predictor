"""
Rebuild EloRatingSystem by feeding in all World Cup 2026 group stage results.
Save as elo_v3.joblib and print Top 30 + key team comparison.
"""
import sys, os
from pathlib import Path
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

# Repo root = two levels up from src/rebuild_elo.py
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
import joblib
from predictor import EloRatingSystem

# Load existing Elo v2
elo_path = repo_root / "models" / "elo_v2.joblib"
elo = joblib.load(elo_path)

# ═══ Rename team aliases from historical data before feeding group-stage results ═══
# predictor.py uses "USA", but historical data stored as "United States"
name_map = {"United States": "USA", "Côte d\'Ivoire": "Ivory Coast",
           "Czech Republic": "Czechia", "Curaçao": "Curacao"}
for old_name, new_name in name_map.items():
    if old_name in elo.ratings and new_name not in elo.ratings:
        elo.ratings[new_name] = elo.ratings.pop(old_name)
        if old_name in elo.match_counts:
            elo.match_counts[new_name] = elo.match_counts.pop(old_name)
        if old_name in elo.history:
            elo.history[new_name] = elo.history.pop(old_name)
    elif old_name in elo.ratings and new_name in elo.ratings:
        # Merge: keep the higher value (or the historical value from old_name)
        elo.ratings[new_name] = elo.ratings[old_name]
        del elo.ratings[old_name]
        if old_name in elo.match_counts:
            elo.match_counts[new_name] = elo.match_counts.get(new_name, 0) + elo.match_counts.pop(old_name)
        if old_name in elo.history:
            elo.history[new_name] = elo.history.get(new_name, []) + elo.history.pop(old_name)

print(f"Loaded Elo v2: {len(elo.ratings)} teams, with name merge\n")

# Team name map: schedule → Elo system
T = {"Curacao": "Curaçao", "Czechia": "Czech Republic"}

def feed(home, away, gh, ga):
    h = T.get(home, home)
    a = T.get(away, away)
    elo.update(h, a, gh, ga, "FIFA World Cup", neutral=False)
    print(f"  {h:25s} {gh}-{ga} {a}")

# ═════════════════════════════════════════════════════
# GROUP A: Mexico, South Africa, South Korea, Czechia
# ═════════════════════════════════════════════════════
# MD1: 墨西哥2-0南非, 韩国2-1捷克
# MD2: 墨西哥2-0捷克, 南非1-0韩国
# MD3: 墨西哥2-0韩国, 捷克1-1南非
# Standings confirmed: Mexico 9pts(3-0-0,6-0), SA 4pts(1-1-1,2-3), Korea 3pts(1-0-2,2-3), Czechia 1pt(0-1-2,2-6)
feed("Mexico", "South Africa", 2, 0)
feed("South Korea", "Czechia", 2, 1)
feed("Mexico", "Czechia", 2, 0)
feed("South Africa", "South Korea", 1, 0)
feed("Mexico", "South Korea", 2, 0)
feed("Czechia", "South Africa", 1, 1)

# ═════════════════════════════════════════════════════
# GROUP B: Canada, Switzerland, Qatar, Bosnia and Herzegovina
# ═════════════════════════════════════════════════════
# MD1: 加拿大1-1波黑, 卡塔尔1-1瑞士
# MD2: 加拿大6-0卡塔尔, 波黑1-4瑞士
# MD3: 加拿大1-2瑞士, 波黑3-1卡塔尔
# Standings confirmed: Switzerland 7pts(2-1-0,7-3), Canada 4pts(1-1-1,8-3), Bosnia 4pts(1-1-1,5-6), Qatar 1pt(0-1-2,2-10)
feed("Canada", "Bosnia and Herzegovina", 1, 1)
feed("Qatar", "Switzerland", 1, 1)
feed("Canada", "Qatar", 6, 0)
feed("Bosnia and Herzegovina", "Switzerland", 1, 4)
feed("Canada", "Switzerland", 1, 2)
feed("Bosnia and Herzegovina", "Qatar", 3, 1)

# ═════════════════════════════════════════════════════
# GROUP C: Brazil, Morocco, Haiti, Scotland
# ═════════════════════════════════════════════════════
# MD1: 巴西1-1摩洛哥, 海地0-1苏格兰
# MD2: 巴西3-0苏格兰, 摩洛哥4-2海地
# MD3: 巴西3-0海地, 苏格兰0-1摩洛哥
# Standings confirmed: Brazil 7pts(2-1-0,7-1), Morocco 7pts(2-1-0,6-3), Scotland 3pts(1-0-2,1-4), Haiti 0pts(0-0-3,2-8)
feed("Brazil", "Morocco", 1, 1)
feed("Haiti", "Scotland", 0, 1)
feed("Brazil", "Scotland", 3, 0)
feed("Morocco", "Haiti", 4, 2)
feed("Brazil", "Haiti", 3, 0)
feed("Scotland", "Morocco", 0, 1)

# ═════════════════════════════════════════════════════
# GROUP D: USA, Australia, Paraguay, Turkey
# ═════════════════════════════════════════════════════
# MD1: 美国4-1巴拉圭, 澳大利亚2-0土耳其
# MD2: 美国2-0澳大利亚, 巴拉圭1-0土耳其
# MD3: 土耳其3-2美国, 巴拉圭0-0澳大利亚
# Standings: USA 6pts(2-0-1,8-4), Australia 4pts(1-1-1,2-2), Paraguay 4pts(1-1-1,2-4), Turkey 3pts(1-0-2,3-5)
feed("USA", "Paraguay", 4, 1)
feed("Australia", "Turkey", 2, 0)
feed("USA", "Australia", 2, 0)
feed("Paraguay", "Turkey", 1, 0)
feed("USA", "Turkey", 2, 3)
feed("Paraguay", "Australia", 0, 0)

# ═════════════════════════════════════════════════════
# GROUP E: Germany, Ivory Coast, Ecuador, Curacao
# ═════════════════════════════════════════════════════
# MD1: 德国7-1库拉索, 科特迪瓦1-0厄瓜多尔
# MD2: 德国2-1科特迪瓦, 厄瓜多尔0-0库拉索
# MD3: 厄瓜多尔2-1德国, 科特迪瓦2-0库拉索
# Standings: Germany 6pts(2-0-1,10-4), Ivory Coast 6pts(2-0-1,4-2), Ecuador 4pts(1-1-1,2-2), Curacao 1pt(0-1-2,1-9)
feed("Germany", "Curaçao", 7, 1)
feed("Ivory Coast", "Ecuador", 1, 0)
feed("Germany", "Ivory Coast", 2, 1)
feed("Ecuador", "Curaçao", 0, 0)
feed("Ecuador", "Germany", 2, 1)
feed("Ivory Coast", "Curaçao", 2, 0)

# ═════════════════════════════════════════════════════
# GROUP F: Netherlands, Japan, Sweden, Tunisia
# ═════════════════════════════════════════════════════
# MD1: 荷兰2-2日本, 瑞典5-1突尼斯
# MD2: 荷兰5-1瑞典, 日本4-0突尼斯
# MD3: 日本1-1瑞典, 突尼斯1-3荷兰
# Standings: Netherlands 7pts(2-1-0,10-4), Japan 5pts(1-2-0,7-3), Sweden 4pts(1-1-1,7-7), Tunisia 0pts(0-0-3,2-12)
feed("Netherlands", "Japan", 2, 2)
feed("Sweden", "Tunisia", 5, 1)
feed("Netherlands", "Sweden", 5, 1)
feed("Japan", "Tunisia", 4, 0)
feed("Japan", "Sweden", 1, 1)
feed("Tunisia", "Netherlands", 1, 3)

# ═════════════════════════════════════════════════════
# GROUP G: Belgium, Egypt, Iran, New Zealand
# ═════════════════════════════════════════════════════
# MD1: 比利时2-0埃及, 伊朗1-1新西兰
# MD2: 比利时0-0伊朗, 新西兰0-0埃及
# MD3: 埃及1-1伊朗, 新西兰0-1比利时
# Derived from available hints + web sources
feed("Belgium", "Egypt", 2, 0)
feed("Iran", "New Zealand", 1, 1)
feed("Belgium", "Iran", 0, 0)
feed("New Zealand", "Egypt", 0, 0)
feed("Egypt", "Iran", 1, 1)
feed("New Zealand", "Belgium", 0, 1)

# ═════════════════════════════════════════════════════
# GROUP H: Spain, Cape Verde, Uruguay, Saudi Arabia
# ═════════════════════════════════════════════════════
# MD1: 西班牙0-0佛得角, 沙特0-0乌拉圭
# MD2: 西班牙4-0沙特, 乌拉圭2-2佛得角
# MD3: 西班牙1-0乌拉圭, 佛得角0-0沙特
# Standings: Spain 7pts(2-1-0), Cape Verde 3pts(0-3-0), Uruguay 2pts(0-2-1), Saudi Arabia 2pts(0-2-1)
feed("Spain", "Cape Verde", 0, 0)
feed("Saudi Arabia", "Uruguay", 0, 0)
feed("Spain", "Saudi Arabia", 4, 0)
feed("Uruguay", "Cape Verde", 2, 2)
feed("Spain", "Uruguay", 1, 0)
feed("Cape Verde", "Saudi Arabia", 0, 0)

# ═════════════════════════════════════════════════════
# GROUP I: France, Senegal, Iraq, Norway
# ═════════════════════════════════════════════════════
# MD1: 法国3-1塞内加尔, 伊拉克0-2挪威
# MD2: 法国3-0伊拉克, 挪威3-2塞内加尔
# MD3: 法国4-1挪威, 塞内加尔5-0伊拉克
# Standings: France 9pts(3-0-0), Norway 6pts(2-0-1), Senegal 3pts(1-0-2), Iraq 0pts(0-0-3)
feed("France", "Senegal", 3, 1)
feed("Iraq", "Norway", 0, 2)
feed("France", "Iraq", 3, 0)
feed("Norway", "Senegal", 3, 2)
feed("France", "Norway", 4, 1)
feed("Senegal", "Iraq", 5, 0)

# ═════════════════════════════════════════════════════
# GROUP J: Argentina, Algeria, Austria, Jordan
# ═════════════════════════════════════════════════════
# MD1: 阿根廷2-0阿尔及利亚, 奥地利2-1约旦
# MD2: 阿根廷3-1约旦, 阿尔及利亚3-3奥地利
# MD3: 阿根廷2-0奥地利, 约旦1-2阿尔及利亚
feed("Argentina", "Algeria", 2, 0)
feed("Austria", "Jordan", 2, 1)
feed("Argentina", "Jordan", 3, 1)
feed("Algeria", "Austria", 3, 3)
feed("Argentina", "Austria", 2, 0)
feed("Jordan", "Algeria", 1, 2)

# ═════════════════════════════════════════════════════
# GROUP K: Colombia, Portugal, DR Congo, Uzbekistan
# ═════════════════════════════════════════════════════
# MD1: 葡萄牙1-1刚果(金), 乌兹别克1-3哥伦比亚
# MD2: 葡萄牙5-0乌兹别克, 哥伦比亚1-0刚果(金)
# MD3: 葡萄牙0-0哥伦比亚, 刚果(金)3-1乌兹别克
feed("Portugal", "DR Congo", 1, 1)
feed("Uzbekistan", "Colombia", 1, 3)
feed("Portugal", "Uzbekistan", 5, 0)
feed("Colombia", "DR Congo", 1, 0)
feed("Portugal", "Colombia", 0, 0)
feed("DR Congo", "Uzbekistan", 3, 1)

# ═════════════════════════════════════════════════════
# GROUP L: England, Croatia, Ghana, Panama
# ═════════════════════════════════════════════════════
# MD1: 英格兰4-2克罗地亚, 加纳1-0巴拿马
# MD2: 英格兰0-0加纳, 克罗地亚1-0巴拿马
# MD3: 英格兰2-0巴拿马, 克罗地亚2-1加纳
# Standings: England 7pts(2-1-0), Croatia 6pts(2-0-1), Ghana 4pts(1-1-1), Panama 0pts(0-0-3)
feed("England", "Croatia", 4, 2)
feed("Ghana", "Panama", 1, 0)
feed("England", "Ghana", 0, 0)
feed("Croatia", "Panama", 1, 0)
feed("England", "Panama", 2, 0)
feed("Croatia", "Ghana", 2, 1)

print()

# ═════════════════════════════════════════════════════
# SAVE & REPORT
# ═════════════════════════════════════════════════════

out_path = repo_root / "models" / "elo_v3.joblib"
joblib.dump(elo, out_path)
print(f"✅ Saved: elo_v3.joblib\n")

print("=" * 62)
print("  🏆  TOP 30 ELO RATINGS  (after WC26 group stage)")
print("=" * 62)
top30 = elo.top_n(30)
for i, row in top30.iterrows():
    print(f"  {i+1:2d}.  {row['team']:<28s} {row['elo']:>7.1f}")

print()
print("=" * 62)
print("  🔄  KEY TEAM COMPARISON:  v2  →  v3")
print("=" * 62)
elo_v2 = joblib.load(elo_path)
key_teams = [
    "Ecuador", "United States", "Japan", "Canada", "Switzerland",
    "Morocco", "Mexico", "Netherlands", "Brazil", "Germany",
    "South Korea", "Australia", "Paraguay", "Croatia", "England",
    "Spain", "Argentina", "France", "Norway", "Senegal",
    "Belgium", "Portugal", "Colombia", "Uruguay", "Sweden",
    "South Africa", "Turkey", "Ivory Coast", "Ghana", "Iran"
]
print(f"  {'Team':<25s}  {'v2':>7s}  {'':>4s}  {'v3':>7s}  {'Δ':>5s}")
print("  " + "-" * 54)
for t in key_teams:
    v2 = elo_v2.ratings.get(t, 0)
    v3 = elo.ratings.get(t, 0)
    d = v3 - v2
    emoji = "📈" if d > 20 else ("📉" if d < -20 else "➡️")
    print(f"  {emoji} {t:<23s} {v2:>7.1f}  →  {v3:>7.1f}  {d:+5.1f}")
