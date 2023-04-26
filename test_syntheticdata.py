##test synthetic data
from synthetic_data import DataGenerator
import matplotlib.pyplot as plt
from UCM import UCM_algorithm


def test_causality():
    N=[50,100,200,300,400,500, 1000,1500,2000,2500,3000]
    sup = [2,3,4,5]
    prob ={k:{} for k in sup}
    for val in sup:
        prob_N = {k:0 for k in N}
        for instances in N:
            times_rightdirection=0
            for i in range(100):
                generator = DataGenerator().getinstances(instances, [val,val])
                data = generator[0]
                while any(len(t)!=val for t in [data[column].unique() for column in list(data.columns)]):
                    generator = DataGenerator().getinstances(instances, [val,val])
                    data = generator[0]
                scores = UCM_algorithm(data).find_best_direction(data, ['no','cyclic'])
                if scores[0]<scores[1]:
                    times_rightdirection+=1
            prob_right = times_rightdirection/100
            print(prob_right)
            prob_N[instances] = prob_right
        prob[val] = prob_N
    return prob

def plot_causality():
    results = test_causality()
    print(results)
    colors = ['#C71585','#1f77b4', 'green', "#ff8533"]
    x = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    sup = [2,3,4,5]
    for support in sup:
        plt.plot(x, list(results[support].values()), marker="o", color = colors[sup.index(support)], label=r'$ |\mathcal{X}| = |\mathcal{Y}| =$' + str(support))
    plt.xlabel(r'$ \mathit{N} $', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', prop={'size': 11})
    plt.show()
    

def test_causality_difsup():
    N=[50,100,200,300,400,500, 1000,1500,2000,2500,3000]
    sup = [[3,2], [2,3], [4,2], [2,4], [5,2], [2,5]]
    prob ={tuple(k):{} for k in sup}
    for val in sup:
        prob_N = {k:0 for k in N}
        for instances in N:
            times_rightdirection=0
            for i in range(100):
                generator = DataGenerator().getinstances(instances, [val[0],val[1]])
                data = generator[0] 
                col = [len(data[column].unique()) for column in list(data.columns)]
                while any(col[t]!=val[t] for t in range(len(col))):
                    generator = DataGenerator().getinstances(instances, [val[0],val[1]])
                    data = generator[0]
                    col = [len(data[column].unique()) for column in list(data.columns)]
                scores = UCM_algorithm(data).find_best_direction(data, ['no','cyclic'])
                if scores[0]<scores[1]:
                    times_rightdirection+=1
            prob_right = times_rightdirection/100
            print(prob_right)
            prob_N[instances] = prob_right
        prob[tuple(val)] = prob_N
    return prob

def plot_causality_difsup():
    results = test_causality_difsup()
    print(results)
    x = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    sup = [[3,2], [2,3], [4,2], [2,4], [5,2], [2,5]]
    for support in sup:
        if support == [3,2] or support == [2,3]:
            col = '#C71585'
        elif support == [4,2] or support == [2,4]: 
            col = '#1f77b4'
        else:
            col = 'green'
        if support == [3,2] or support == [4,2] or support == [5,2]:
            plt.plot(x, list(results[tuple(support)].values()), marker="o", color = col, linestyle='dashed', label=r'$ |\mathcal{X}| = $' + str(support[0]) + r'$, |\mathcal{Y}| =$' + str(support[1]))
        else:
            plt.plot(x, list(results[tuple(support)].values()), marker="o", color = col, label=r'$ |\mathcal{X}| =$' + str(support[0]) + r'$, |\mathcal{Y}| =$' + str(support[1]))
    plt.xlabel(r'$ \mathit{N} $', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', prop={'size': 11})
    plt.show()



print(plot_causality_difsup())

