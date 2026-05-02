from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Intervalo de theta y x
a, b = 0, 1

# Media y desviación estándar de grupo 1
mu_g1 = 0.7
std_g1 = 0.15
a_g1 = (a - mu_g1) / std_g1
b_g1 = (b - mu_g1) / std_g1

# Media y desviación estándar de grupo 2
mu_g2 = 0.3
std_g2 = 0.15
a_g2 = (a - mu_g2) / std_g2
b_g2 = (b - mu_g2) / std_g2

# Añado grupo 3 con su media y desviación estándar
mu_g3 = 0.5
std_g3 = 0.15
a_g3 = (a - mu_g3) / std_g3
b_g3 = (b - mu_g3) / std_g3

# añado cont_d0 para el coupled sampling individual
class User():
    def __init__(self, user_id, group, theta, x, cont_d0=0):
        self.user_id = user_id
        self.group = group
        self.theta = theta
        self.x = x
        self.cont_d0 = cont_d0
        
def train_LR(X, y):
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(X, y)  
    return model 

def create_user_homophily(id, n_g1, n_g2):

    p_g1 = n_g1 / (n_g1 + n_g2)
    p = np.random.binomial(1, p_g1)

    if p == 1:
        group = 1
        theta = truncnorm.rvs(a_g1, b_g1, loc=mu_g1, scale=std_g1)
    else:
        group = 2
        theta = truncnorm.rvs(a_g2, b_g2, loc=mu_g2, scale=std_g2)

    x = theta
    return User(id, group, theta, x)

def plot_user_checkpoints(g1_T, g2_T, checkpoints):
    grupo1 = np.array(g1_T)
    grupo2 = np.array(g2_T)

    checkpoints_validos = [i for i in checkpoints if i < len(grupo1) and i < len(grupo2)]

    g1 = grupo1[checkpoints_validos]
    g2 = grupo2[checkpoints_validos]

    x = np.arange(len(checkpoints_validos))
    width = 0.34

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"
    c2 = "#2A6F97"

    ax.bar(
        x - width/2, g1, width,
        color=c1,
        alpha=0.85,
        edgecolor=c1,
        linewidth=1.6,
        label="Group 1"
    )

    ax.bar(
        x + width/2, g2, width,
        color=c2,
        alpha=0.85,
        edgecolor=c2,
        linewidth=1.6,
        label="Group 2"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in checkpoints_validos])
    ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
    ax.set_ylabel("Number of users", fontsize=22, fontfamily="serif")

    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.set_ylim(0, 1000)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


# para graficar boxplots (del interes o del error). En legend_loc poner
# posición de la leyenda, si se pone nada, entonces por defecto toma valor
# "lower right". En concepto poner "theta" o "error", en función del boxplot
# deseado. 
def plot_boxplots(theta_T_g1, theta_T_g2, checkpoints, concept, legend_loc="lower right", legend_ncol=2):
    data_g1 = [theta_T_g1[t] for t in checkpoints]
    data_g2 = [theta_T_g2[t] for t in checkpoints]

    pos_g1 = [i - 0.18 for i in range(len(checkpoints))]
    pos_g2 = [i + 0.18 for i in range(len(checkpoints))]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"
    c2 = "#2A6F97"

    ax.boxplot(
        data_g1,
        positions=pos_g1,
        widths=0.28,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=c1, alpha=0.15, edgecolor=c1, linewidth=2.0),
        medianprops=dict(color=c1, linewidth=2.2),
        whiskerprops=dict(color=c1, linewidth=1.8),
        capprops=dict(color=c1, linewidth=1.8),
        flierprops=dict(
            marker='o',
            markerfacecolor='none',
            markeredgecolor=c1,
            markeredgewidth=1.5,
            markersize=4.8,
            linestyle='none'
        )
    )

    ax.boxplot(
        data_g2,
        positions=pos_g2,
        widths=0.28,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=c2, alpha=0.15, edgecolor=c2, linewidth=2.0),
        medianprops=dict(color=c2, linewidth=2.2),
        whiskerprops=dict(color=c2, linewidth=1.8),
        capprops=dict(color=c2, linewidth=1.8),
        flierprops=dict(
            marker='o',
            markerfacecolor='none',
            markeredgecolor=c2,
            markeredgewidth=1.5,
            markersize=4.8,
            linestyle='none'
        )
    )

    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([f"{x:,}" for x in checkpoints])
    
    if concept == "theta":
        ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
        ax.set_ylabel(r"$\theta$", fontsize=22, fontfamily="serif")
        ax.axhline(0.5, color='0.35', linestyle='--', linewidth=1.2, dashes=(4, 4))
    elif concept == "error":
        ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
        ax.set_ylabel(r"$\hat{y} - \mathbb{E}[y]$", fontsize=22, fontfamily="serif")
        ax.axhline(0, color='0.35', linestyle='--', linewidth=1.2, dashes=(4, 4))
    elif concept == "error_outcomefl":
        ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
        ax.set_ylabel(r"$\hat{y} - \theta$", fontsize=22, fontfamily="serif")
        ax.axhline(0, color='0.35', linestyle='--', linewidth=1.2, dashes=(4, 4))
        


    legend_handles = [
        Patch(facecolor=c1, edgecolor=c1, alpha=0.15, linewidth=2.0, label='Group 1'),
        Patch(facecolor=c2, edgecolor=c2, alpha=0.15, linewidth=2.0, label='Group 2')
    ]
    ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        ncol=legend_ncol,
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def plot_theta_histogram(population):
    groups = sorted(set(user.group for user in population))
    if len(groups) != 2:
        raise ValueError("La función está pensada para exactamente 2 grupos.")

    g1_label, g2_label = groups[0], groups[1]

    theta_g1 = np.array([user.theta for user in population if user.group == g1_label])
    theta_g2 = np.array([user.theta for user in population if user.group == g2_label])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"
    c2 = "#2A6F97"

    ax.hist(
        theta_g1,
        bins=20,
        color=c1,
        alpha=0.55,
        edgecolor=c1,
        linewidth=1.4,
        label="Group 1"
    )

    ax.hist(
        theta_g2,
        bins=20,
        color=c2,
        alpha=0.55,
        edgecolor=c2,
        linewidth=1.4,
        label="Group 2"
    )

    ax.set_xlabel(r"$\theta$", fontsize=22, fontfamily="serif")
    ax.set_ylabel("Number of users", fontsize=22, fontfamily="serif")

    ax.legend(
        loc="upper right",
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
# Función que calcula media de un array
def mean(arr):
    sumador = 0
    for i in arr:
        sumador += i
    return sumador/len(arr)

# Cálculo de E(y)
def error(population, model):
    err_g1 = []
    err_g2 = []
    for i in population:
        outcomes = []
        theta = i.theta
        y_hat = model.predict_proba([[i.x]])[0,1]   #devuelve probabilidad de la clase 1
        for j in range(50):
            p = truncnorm.rvs((0-theta)/0.1, (1-theta)/0.1, loc=theta, scale=0.1) 
            outcome = np.random.binomial(1, p)
            outcomes.append(outcome)
        y_avg = mean(outcomes)
        if i.group == 1:
            err_g1.append(y_hat - y_avg)
        else:
            err_g2.append(y_hat - y_avg)
    return (err_g1, err_g2)

def error_outcomefl(population, model):
    err_g1 = []
    err_g2 = []
    for i in population:
        outcomes = []
        theta = i.theta
        y_hat = model.predict_proba([[i.x]])[0,1]   #devuelve probabilidad de la clase 1
        if i.group == 1:
            err_g1.append(y_hat - theta)
        else:
            err_g2.append(y_hat - theta)
    return (err_g1, err_g2)




   
# para graficar numero de usuarios por time step, ahora como
# curva continua, en lugar de como barras 
def plot_user_T(g1_T, g2_T):
    grupo1 = np.array(g1_T)
    grupo2 = np.array(g2_T)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"
    c2 = "#2A6F97"

    ax.plot(grupo1, color=c1, linewidth=2.2, label="Group 1")
    ax.plot(grupo2, color=c2, linewidth=2.2, label="Group 2")

    ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
    ax.set_ylabel("Number of users", fontsize=22, fontfamily="serif")

    ax.legend(
        loc="best",
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
    
    





# 3 group functions    
def create_user_homophily_3groups(id, n_g1, n_g2, n_g3):

    total = n_g1 + n_g2 + n_g3
    probs = [n_g1/total, n_g2/total, n_g3/total]

    group = np.random.choice([1,2,3], p=probs)

    if group == 1:
        theta = truncnorm.rvs(a_g1, b_g1, loc=mu_g1, scale=std_g1)
    elif group == 2:
        theta = truncnorm.rvs(a_g2, b_g2, loc=mu_g2, scale=std_g2)
    else:
        theta = truncnorm.rvs(a_g3, b_g3, loc=mu_g3, scale=std_g3)

    x = theta
    return User(id, group, theta, x)

import numpy as np
import matplotlib.pyplot as plt

def plot_theta_histogram_3groups(population):
    groups = sorted(set(user.group for user in population))
    if len(groups) != 3:
        raise ValueError("La función está pensada para exactamente 3 grupos.")

    g1_label, g2_label, g3_label = groups[0], groups[1], groups[2]

    theta_g1 = np.array([user.theta for user in population if user.group == g1_label])
    theta_g2 = np.array([user.theta for user in population if user.group == g2_label])
    theta_g3 = np.array([user.theta for user in population if user.group == g3_label])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"   # granate
    c2 = "#2A6F97"   # azul petróleo
    c3 = "#C17C00"   # mostaza / ocre

    ax.hist(
        theta_g1,
        bins=20,
        color=c1,
        alpha=0.45,
        edgecolor=c1,
        linewidth=1.4,
        label="Group 1"
    )

    ax.hist(
        theta_g2,
        bins=20,
        color=c2,
        alpha=0.45,
        edgecolor=c2,
        linewidth=1.4,
        label="Group 2"
    )

    ax.hist(
        theta_g3,
        bins=20,
        color=c3,
        alpha=0.45,
        edgecolor=c3,
        linewidth=1.4,
        label="Group 3"
    )

    ax.set_xlabel(r"$\theta$", fontsize=22, fontfamily="serif")
    ax.set_ylabel("Number of users", fontsize=22, fontfamily="serif")

    ax.legend(
        loc="upper right",
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

def plot_user_checkpoints_3groups(g1_T, g2_T, g3_T, checkpoints):
    grupo1 = np.array(g1_T)
    grupo2 = np.array(g2_T)
    grupo3 = np.array(g3_T)

    checkpoints_validos = [
        i for i in checkpoints
        if i < len(grupo1) and i < len(grupo2) and i < len(grupo3)
    ]

    g1 = grupo1[checkpoints_validos]
    g2 = grupo2[checkpoints_validos]
    g3 = grupo3[checkpoints_validos]

    x = np.arange(len(checkpoints_validos))
    width = 0.24

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"
    c2 = "#2A6F97"
    c3 = "#C17C00"

    ax.bar(
        x - width, g1, width,
        color=c1,
        alpha=0.85,
        edgecolor=c1,
        linewidth=1.6,
        label="Group 1"
    )

    ax.bar(
        x, g2, width,
        color=c2,
        alpha=0.85,
        edgecolor=c2,
        linewidth=1.6,
        label="Group 2"
    )

    ax.bar(
        x + width, g3, width,
        color=c3,
        alpha=0.85,
        edgecolor=c3,
        linewidth=1.6,
        label="Group 3"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in checkpoints_validos])
    ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
    ax.set_ylabel("Number of users", fontsize=22, fontfamily="serif")

    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.set_ylim(0, 1000)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
    # para graficar boxplots (del interes o del error). En legend_loc poner
# posición de la leyenda, si se pone nada, entonces por defecto toma valor
# "lower right". En concept poner "theta" o "error", en función del boxplot
# deseado.
def plot_boxplots_3groups(theta_T_g1, theta_T_g2, theta_T_g3, checkpoints, concept, legend_loc="lower right"):
    data_g1 = [theta_T_g1[t] for t in checkpoints]
    data_g2 = [theta_T_g2[t] for t in checkpoints]
    data_g3 = [theta_T_g3[t] for t in checkpoints]

    pos_g1 = [i - 0.24 for i in range(len(checkpoints))]
    pos_g2 = [i for i in range(len(checkpoints))]
    pos_g3 = [i + 0.24 for i in range(len(checkpoints))]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"
    c2 = "#2A6F97"
    c3 = "#C17C00"

    ax.boxplot(
        data_g1,
        positions=pos_g1,
        widths=0.20,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=c1, alpha=0.15, edgecolor=c1, linewidth=2.0),
        medianprops=dict(color=c1, linewidth=2.2),
        whiskerprops=dict(color=c1, linewidth=1.8),
        capprops=dict(color=c1, linewidth=1.8),
        flierprops=dict(
            marker='o',
            markerfacecolor='none',
            markeredgecolor=c1,
            markeredgewidth=1.5,
            markersize=4.8,
            linestyle='none'
        )
    )

    ax.boxplot(
        data_g2,
        positions=pos_g2,
        widths=0.20,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=c2, alpha=0.15, edgecolor=c2, linewidth=2.0),
        medianprops=dict(color=c2, linewidth=2.2),
        whiskerprops=dict(color=c2, linewidth=1.8),
        capprops=dict(color=c2, linewidth=1.8),
        flierprops=dict(
            marker='o',
            markerfacecolor='none',
            markeredgecolor=c2,
            markeredgewidth=1.5,
            markersize=4.8,
            linestyle='none'
        )
    )

    ax.boxplot(
        data_g3,
        positions=pos_g3,
        widths=0.20,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=c3, alpha=0.15, edgecolor=c3, linewidth=2.0),
        medianprops=dict(color=c3, linewidth=2.2),
        whiskerprops=dict(color=c3, linewidth=1.8),
        capprops=dict(color=c3, linewidth=1.8),
        flierprops=dict(
            marker='o',
            markerfacecolor='none',
            markeredgecolor=c3,
            markeredgewidth=1.5,
            markersize=4.8,
            linestyle='none'
        )
    )

    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([f"{x:,}" for x in checkpoints])

    if concept == "theta":
        ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
        ax.set_ylabel(r"$\theta$", fontsize=22, fontfamily="serif")
        ax.axhline(0.5, color='0.35', linestyle='--', linewidth=1.2, dashes=(4, 4))
    elif concept == "error":
        ax.set_xlabel("Time steps", fontsize=22, fontfamily="serif")
        ax.set_ylabel(r"$\hat{y} - \mathbb{E}[y]$", fontsize=22, fontfamily="serif")
        ax.axhline(0, color='0.35', linestyle='--', linewidth=1.2, dashes=(4, 4))

    legend_handles = [
        Patch(facecolor=c1, edgecolor=c1, alpha=0.15, linewidth=2.0, label='Group 1'),
        Patch(facecolor=c2, edgecolor=c2, alpha=0.15, linewidth=2.0, label='Group 2'),
        Patch(facecolor=c3, edgecolor=c3, alpha=0.15, linewidth=2.0, label='Group 3')
    ]
    ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        ncol=3,
        frameon=False,
        prop={"family": "serif", "size": 18}
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    

def error_3groups(population, model):
    err_g1 = []
    err_g2 = []
    err_g3 = []
    for i in population:
        outcomes = []
        theta = i.theta
        y_hat = model.predict_proba([[i.x]])[0,1]   #devuelve probabilidad de la clase 1
        for j in range(50):
            p = truncnorm.rvs((0-theta)/0.1, (1-theta)/0.1, loc=theta, scale=0.1) 
            outcome = np.random.binomial(1, p)
            outcomes.append(outcome)
        y_avg = mean(outcomes)
        if i.group == 1:
            err_g1.append(y_hat - y_avg)
        elif i.group ==2:
            err_g2.append(y_hat - y_avg)
        else:
            err_g3.append(y_hat - y_avg)
    return (err_g1, err_g2, err_g3)




# Outcome Feedback Loop
def sample_base_outcome(theta, std=0.1):
    p = truncnorm.rvs((0-theta)/std, (1-theta)/std, loc=theta, scale=std)
    return np.random.binomial(1, p)

def sample_modified_outcome(theta, d, outcome_bias=0.2, std=0.05):
    """
    Aquí sigo tu convención actual: re-muestreo usando theta.
    Si luego quieres, lo cambias por x.
    """
    if d == 0:
        p = truncnorm.rvs((0-(theta-outcome_bias))/std,
                          (1-(theta-outcome_bias))/std,
                          loc=theta-outcome_bias, scale=std)
    else:  # d == 1
        p = truncnorm.rvs((0-(theta+outcome_bias))/std,
                          (1-(theta+outcome_bias))/std,
                          loc=theta+outcome_bias, scale=std)
    return np.random.binomial(1, p)

def estimate_plot_curves(population, model, x_grid=None, n_rep=200, bin_width=0.025):
    
    #Devuelve todo lo necesario para graficar:
    #- curva predictor final
    #- curva P[t(theta)=1 | x]
    #- curva P[t(theta,d)=1 | x]
    
    if x_grid is None:
        x_grid = np.linspace(0, 1, 201)

    # Curva del predictor final
    y_pred_final = model.predict_proba(x_grid.reshape(-1, 1))[:, 1]

    # Para estimar las dos curvas "verdaderas"
    xs = []
    base_means = []
    mod_means = []

    for user in population:
        x = user.x
        theta = user.theta

        # predicción dura para decidir d, como en tu flujo
        d = model.predict([[x]])[0]

        base_samples = [sample_base_outcome(theta) for _ in range(n_rep)]
        mod_samples  = [sample_modified_outcome(theta, d) for _ in range(n_rep)]

        xs.append(x)
        base_means.append(np.mean(base_samples))
        mod_means.append(np.mean(mod_samples))

    xs = np.array(xs)
    base_means = np.array(base_means)
    mod_means = np.array(mod_means)

    # Agrupar por bins de x para obtener curvas suaves
    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    y_base_curve = []
    y_mod_curve = []

    for left, right in zip(bins[:-1], bins[1:]):
        mask = (xs >= left) & (xs < right)
        if np.any(mask):
            y_base_curve.append(np.mean(base_means[mask]))
            y_mod_curve.append(np.mean(mod_means[mask]))
        else:
            y_base_curve.append(np.nan)
            y_mod_curve.append(np.nan)

    return {
        "x_grid": x_grid,
        "y_pred_final": y_pred_final,
        "bin_centers": bin_centers,
        "y_base_curve": np.array(y_base_curve),
        "y_mod_curve": np.array(y_mod_curve),
    }


def get_initial_predict_curve(model_init, x_grid=None):
    if x_grid is None:
        x_grid = np.linspace(0, 1, 201)
    y_pred_init = model_init.predict_proba(x_grid.reshape(-1, 1))[:, 1]
    return x_grid, y_pred_init

def plot_curves(plot_data, A_init=None):
    X_init = plot_data["X_init"].ravel()
    y_init = plot_data["y_init"]

    x_grid = plot_data["x_grid"]
    y_pred_init = plot_data["y_pred_init"]
    y_pred_final = plot_data["y_pred_final"]

    bin_centers = plot_data["bin_centers"]
    y_base_curve = plot_data["y_base_curve"]
    y_mod_curve = plot_data["y_mod_curve"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 18
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)

    c1 = "#8E3B46"   
    c2 = "#2A6F97"   
    c3 = "#C17C00"   

    # Scatter inicial
    if A_init is not None:
        A_init = np.array(A_init)
        mask_g1 = (A_init == 1)
        mask_g2 = (A_init == 2)

        ax.scatter(
            X_init[mask_g1], y_init[mask_g1],
            s=80, alpha=0.55, color=c1, edgecolors="none"
        )
        ax.scatter(
            X_init[mask_g2], y_init[mask_g2],
            s=80, alpha=0.55, color=c2, edgecolors="none"
        )
    else:
        ax.scatter(
            X_init, y_init,
            s=80, alpha=0.55, color="black", edgecolors="none"
        )

    # Curvas
    line_base, = ax.plot(
        bin_centers, y_base_curve,
        linestyle="--", dashes=(5, 5),
        linewidth=2.0, color="black",
        label=r"$P[t(\theta)=1 \mid r(\theta)=x]$"
    )

    line_mod, = ax.plot(
        bin_centers, y_mod_curve,
        linestyle="--", dashes=(7, 4),
        linewidth=2.0, color=c3,
        label=r"$P[t(\theta,d)=1 \mid r(\theta)=x]$"
    )

    line_init, = ax.plot(
        x_grid, y_pred_init,
        linewidth=2.2, color="black",
        label="Initial predict."
    )

    line_final, = ax.plot(
        x_grid, y_pred_final,
        linewidth=2.2, color=c3,
        label="Final predict."
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    #ax.margins(x=0, y=0)

    ax.set_xlabel(r"$x$", fontsize=22, fontfamily="serif")
    ax.set_ylabel("y", fontsize=22, fontfamily="serif")

    # Leyenda 1: curvas
    legend_curves = ax.legend(
        handles=[line_base, line_mod, line_init, line_final],
        loc="upper left",
        frameon=False,
        prop={"family": "serif", "size": 16},
        handlelength=3.8,
        handletextpad=0.5
    )
    ax.add_artist(legend_curves)

    # Leyenda 2: grupos
    if A_init is not None:
        group_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=c1, markeredgecolor='none',
                   alpha=0.55, markersize=12, label='Group 1 - Initial'),
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=c2, markeredgecolor='none',
                   alpha=0.55, markersize=12, label='Group 2 - Initial')
        ]

        ax.legend(
            handles=group_handles,
            loc="lower right",
            frameon=False,
            prop={"family": "serif", "size": 16},
            handlelength=0,
            handletextpad=0.8
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.2)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
    
    #para el acoplado (P(t(theta... respecto de poblacion inicial)))

def estimate_base_curve(population, n_rep=200, bin_width=0.025):
    xs = []
    base_means = []

    for user in population:
        x = user.x
        theta = user.theta

        base_samples = [sample_base_outcome(theta) for _ in range(n_rep)]
        xs.append(x)
        base_means.append(np.mean(base_samples))

    xs = np.array(xs)
    base_means = np.array(base_means)

    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    y_base_curve = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (xs >= left) & (xs < right)
        if np.any(mask):
            y_base_curve.append(np.mean(base_means[mask]))
        else:
            y_base_curve.append(np.nan)

    return bin_centers, np.array(y_base_curve)

def estimate_plot_curves2(population, model, x_grid=None, n_rep=200, bin_width=0.025):
    if x_grid is None:
        x_grid = np.linspace(0, 1, 201)

    y_pred_final = model.predict_proba(x_grid.reshape(-1, 1))[:, 1]

    xs = []
    mod_means = []

    for user in population:
        x = user.x
        theta = user.theta
        d = model.predict([[x]])[0]

        mod_samples = [sample_modified_outcome(theta, d) for _ in range(n_rep)]

        xs.append(x)
        mod_means.append(np.mean(mod_samples))

    xs = np.array(xs)
    mod_means = np.array(mod_means)

    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    y_mod_curve = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (xs >= left) & (xs < right)
        if np.any(mask):
            y_mod_curve.append(np.mean(mod_means[mask]))
        else:
            y_mod_curve.append(np.nan)

    return {
        "x_grid": x_grid,
        "y_pred_final": y_pred_final,
        "bin_centers_mod": bin_centers,
        "y_mod_curve": np.array(y_mod_curve),
    }