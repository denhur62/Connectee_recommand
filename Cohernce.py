import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

start = 10
limit = 100
step = 10
x = ['2021-09-07', '2021-09-14', '2021-09-21', '2021-09-28']
coherence_values = [0.48234859848,
                    0.52234859848, 0.49234859848, 0.533209471236]

plt.plot(x, coherence_values)
plt.xlabel("date")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
