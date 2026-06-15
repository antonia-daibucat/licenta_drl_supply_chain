import matplotlib.pyplot as plt

# Datele extrase din numele fișierelor best antrenate de mine
pasi_antrenare = [1000, 5000, 31000, 84000, 140000]
recompense = [2.700, 3.450, 4.800, 5.000, 5.200]

plt.figure(figsize=(10, 6))

# Plotăm punctele de control (checkpoints)
plt.plot(pasi_antrenare, recompense, marker='o', color='green', linewidth=2, label='Performanță A2C (Regional)')


plt.title('Evoluția Performanței Agentului A2C pe Nivelul Regional', fontsize=14)
plt.xlabel('Număr de Pași de Antrenare (Steps)', fontsize=12)
plt.ylabel('Recompensă Medie (Reward)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Adăugăm etichete pe puncte pentru claritate
for i, txt in enumerate(recompense):
    plt.annotate(f"+{txt}", (pasi_antrenare[i], recompense[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('evolutie_agent_regional.png', dpi=300)
plt.show()
