import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

data_llm = {
    2018: 477,
    2019: 430,
    2020: 498,
    2021: 520,
    2022: 595,
    2023: 1020,
    2024: 2350
}

data_dl = {
    2018: 8260,
    2019: 8800,
    2020: 9130,
    2021: 10500,
    2022: 11300,
    2023: 11200,
    2024: 12200
}

years = sorted(data_llm)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot LLM data (left axis)
llm_values = [data_llm[y] for y in years]
llm_line, = ax1.plot(years, llm_values, marker='o', color='blue', label="Concept Drift + LLM")
ax1.set_ylabel("LLM Papers", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Estimate ChatGPT release point on LLM line
chatgpt_x = 2022 + 11/12
chatgpt_y = data_llm[2022] + (data_llm[2023] - data_llm[2022]) * (11/12)

# Plot red marker on LLM line
ax1.plot(chatgpt_x, chatgpt_y, 'o', color='red')
ax1.annotate('ChatGPT Release\n(Nov 2022)',
             xy=(chatgpt_x, chatgpt_y),
             xytext=(chatgpt_x + 0.2, chatgpt_y + 300),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# Annotate final LLM point
ax1.annotate(f"{data_llm[2024]}", xy=(2024, data_llm[2024]), xytext=(2024, data_llm[2024] + 300),
             ha='center', fontsize=9, color='blue')

# Plot DL data (right axis)
ax2 = ax1.twinx()
dl_values = [data_dl[y] for y in years]
dl_line, = ax2.plot(years, dl_values, marker='o', color='orange', label="Concept Drift + DL")
ax2.set_ylabel("DL Papers", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Annotate final DL point
ax2.annotate(f"{data_dl[2024]}", xy=(2024, data_dl[2024]), xytext=(2024, data_dl[2024] + 500),
             ha='center', fontsize=9, color='orange')

# Add shared legend (merge handles from both axes)
lines = [llm_line, dl_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True)

# Axis & title
ax1.set_xlim(2018, 2024)
ax1.set_xticks(range(2018, 2025))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title("Concept Drift Papers by Query (2018–2024)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("google_scholar.png", dpi=300)
plt.show()
