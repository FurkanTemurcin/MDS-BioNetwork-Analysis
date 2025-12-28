# -*- coding: utf-8 -*-
"""
MDS_Network_Analysis_Final_Optimized.py
PROJE: Biyolojik Ağlarda Minimum Dominating Set (MDS) Analizi
DERS: İleri Veri Madenciliği

"""

import pandas as pd
import networkx as nx
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time

# --- AYARLAR ---
DOSYA_YOLU = r"C:\Users\FURKAN\OneDrive\Masaüstü\MDS_Proje\veri.tsv"
KAYIT_KLASORU = os.path.dirname(DOSYA_YOLU) 
GUVEN_SKORU = 0.4 

print(f">>> ANALİZ BAŞLATILIYOR...")
baslangic_zamani = time.time()

# =============================================================================
# BÖLÜM 1: VERİ VE AĞ İNŞASI (HIZLI YÜKLEME)
# =============================================================================
try:
    # Veriyi oku
    df = pd.read_csv(DOSYA_YOLU, sep='\t')
    df.columns = df.columns.str.strip()
    
    # Filtreleme (Vektörel işlem - Döngüsüz)
    skor_sutunu = df.columns[-1] 
    df_filtered = df[df[skor_sutunu] >= GUVEN_SKORU]
    
    # Graf oluştur
    G = nx.from_pandas_edgelist(df_filtered, 
                                source=df.columns[0], 
                                target=df.columns[1])
    
    # En büyük parçayı al
    if len(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    print(f"ADIM 1: Veri Yüklendi ({time.time() - baslangic_zamani:.2f} sn)")
    print(f"   -> Düğüm Sayısı: {G.number_of_nodes()}")
    print(f"   -> Kenar Sayısı: {G.number_of_edges()}")

except Exception as e:
    print(f"!!! HATA: {e}")
    exit()

# =============================================================================
# BÖLÜM 2: MDS ALGORİTMASI (ILP)
# =============================================================================
print("\nADIM 2: MDS Hesaplanıyor...")

prob = pulp.LpProblem("MDS", pulp.LpMinimize)
nodes = list(G.nodes())
x = pulp.LpVariable.dicts("x", nodes, cat=pulp.LpBinary)

# Amaç: Minimizasyon
prob += pulp.lpSum([x[n] for n in nodes])

# Kısıt: Kapsama
for n in nodes:
    prob += x[n] + pulp.lpSum([x[neighbor] for neighbor in G.neighbors(n)]) >= 1

prob.solve(pulp.PULP_CBC_CMD(msg=False))
mds_nodes = {n for n in nodes if pulp.value(x[n]) == 1} # Set yapısı
mds_size = len(mds_nodes)

print(f"   -> MDS Boyutu: {mds_size}")

# =============================================================================
# BÖLÜM 3: HUB ve KÜME ANALİZİ
# =============================================================================
print("\nADIM 3: İstatistikler...")
degree_dict = dict(G.degree())
sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
top_k = max(5, int(len(nodes) * 0.10))
hubs = {n[0] for n in sorted_nodes[:top_k]}

# Kümeler
intersect_nodes = mds_nodes.intersection(hubs) # Mor
hidden_criticals = mds_nodes - hubs            # Kırmızı 
pure_hubs = hubs - mds_nodes                   # Mavi

# =============================================================================
# BÖLÜM 4: GÖRSELLEŞTİRME 
# =============================================================================
print("\nADIM 4: Grafikler çiziliyor...")

# Yerleşim hesaplama 
pos = nx.spring_layout(G, seed=42, k=0.55, iterations=60)

def kaydet_goster(isim, title):
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    tam_yol = os.path.join(KAYIT_KLASORU, isim)
    plt.savefig(tam_yol, dpi=300, bbox_inches='tight')
    print(f"   -> Kaydedildi: {isim}")
    plt.show()

# --- GRAFİK 1: HAM AĞ ---
plt.figure(figsize=(12, 10))
# Düğüm: Gri, Küçük (30)
nx.draw_networkx_nodes(G, pos, node_color='#CCCCCC', node_size=30)
# Kenar: Çok silik (%5 görünürlük)
nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='#555555')
kaydet_goster("1_Ham_Ag.png", "1. Ham Protein Ağı")

# --- GRAFİK 2: HUB ANALİZİ ---
plt.figure(figsize=(12, 10))
cols = ['#0000FF' if n in hubs else '#EEEEEE' for n in G.nodes()]
# BOYUT OPTİMİZASYONU: Hub ise 250, değilse 30 (Sabit)
sizes = [250 if n in hubs else 30 for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=cols, node_size=sizes)
nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='#555555')
# Etiket sadece Hub'a
nx.draw_networkx_labels(G, pos, {n:n for n in hubs}, font_size=8, font_color='black', font_weight='bold')
plt.legend(handles=[mpatches.Patch(color='#0000FF', label='Hub (Merkezi)')], loc='upper right')
kaydet_goster("2_Hub_Analizi.png", "2. Klasik Yaklaşım: Hub Analizi")

# --- GRAFİK 3: MDS SONUCU ---
plt.figure(figsize=(12, 10))
cols = ['#FF0000' if n in mds_nodes else '#EEEEEE' for n in G.nodes()]
# BOYUT: MDS ise 250, değilse 30
sizes = [250 if n in mds_nodes else 30 for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=cols, node_size=sizes, edgecolors='black', linewidths=0.5)
nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='#555555')
nx.draw_networkx_labels(G, pos, {n:n for n in mds_nodes}, font_size=9, font_weight='bold')
plt.legend(handles=[mpatches.Patch(color='#FF0000', label='MDS (Kritik)')], loc='upper right')
kaydet_goster("3_MDS_Sonucu.png", f"3. MDS Yöntemi ({mds_size} Gen)")

# --- GRAFİK 4: FİNAL KARŞILAŞTIRMA ---
plt.figure(figsize=(14, 12))
final_cols = []
final_sizes = []

for n in G.nodes():
    if n in intersect_nodes:   
        final_cols.append('purple')
        final_sizes.append(300)
    elif n in hidden_criticals: 
        final_cols.append('red')
        final_sizes.append(300) 
    elif n in pure_hubs:        
        final_cols.append('blue')
        final_sizes.append(150) 
    else:
        final_cols.append('#F5F5F5') 
        final_sizes.append(20) 

nx.draw_networkx_nodes(G, pos, node_color=final_cols, node_size=final_sizes, edgecolors='black', linewidths=0.5)
nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='#333333')

# Etiketler (Çakışmayı önlemek için sadece seçili olanlar)
labels = {n:n for n in G.nodes() if n in mds_nodes or n in hubs}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

# Lejant
L1 = mpatches.Patch(color='purple', label=f'Hem Hub Hem MDS ({len(intersect_nodes)})')
L2 = mpatches.Patch(color='red', label=f'GİZLİ KRİTİK (Sadece MDS) ({len(hidden_criticals)})')
L3 = mpatches.Patch(color='blue', label=f'Sadece Hub ({len(pure_hubs)})')
plt.legend(handles=[L1, L2, L3], loc='upper right', fontsize=11)

kaydet_goster("4_Final_Karsilastirma.png", "4. Detaylı Karşılaştırma")

# =============================================================================
# SUNUM RAPORU
# =============================================================================
print("\n" + "="*40)
print("     SUNUM İÇİN ÖZET RAPORU")
print("="*40)
print(f"Analiz Süresi: {time.time() - baslangic_zamani:.2f} saniye")
print(f"Toplam Gen: {len(nodes)} | Toplam Etkileşim: {G.number_of_edges()}")
print(f"MDS Boyutu: {len(mds_nodes)} (Ağın %{len(mds_nodes)/len(nodes)*100:.1f}'si)")
print("-" * 40)
print(f"1. Mor (Hub+MDS): {len(intersect_nodes)}")
print(f"2. Kırmızı (GİZLİ KRİTİK): {len(hidden_criticals)} <-- Makale Kanıtı")
print("-" * 40)
print(f"Örnek Gizli Genler: {list(hidden_criticals)[:5]}")
print("="*40)