import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(y, t, beta, sigma, gamma, mu):
    """
    Modelo SEIR com Demografia

    Este script implementa um modelo epidemiológico SEIR (Suscetíveis-Expostos-Infectados-Removidos)
    que leva em consideração a demografia (nascimento e morte). As equações diferenciais
    são resolvidas numericamente usando a função odeint do módulo scipy.

    Parâmetros:
        y : list
            Lista que contém as condições iniciais [S, E, I, R].
        t : array
            Vetor de tempo para o qual as equações diferenciais serão resolvidas.
        beta : float
            Taxa de contato (transmissão).
        sigma : float
            Taxa de exposição.
        gamma : float
            Taxa de recuperação.
        mu : float
            Taxa de nascimento/morte.

    Retorna:
        list
            Lista das derivadas das variáveis de estado [dSdt, dEdt, dIdt, dRdt].

    Parâmetros do Modelo:
        - S : Fração da população suscetível.
        - E : Fração da população exposta.
        - I : Fração da população infectada.
        - R : Fração da população removida.

    Condições Iniciais:
        - S0 : Fração inicial de suscetíveis.
        - E0 : Fração inicial de expostos.
        - I0 : Fração inicial de infectados.
        - R0 : Fração inicial de removidos.

    Vetor de Tempo:
        - t : Vetor de tempo para o qual as equações diferenciais são resolvidas
    """    
    
    S, E, I, R = y
    dSdt = mu - beta * S * I - mu * S
    dEdt = beta * S * I - (sigma + mu) * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I - mu * R
    return [dSdt, dEdt, dIdt, dRdt]

# Parâmetros do modelo SEIR com demografia, Rio de Janeiro, 2019
beta = 0.4091   # Taxa de contato (transmissão)
sigma = 0.1818  # Taxa de exposição
gamma = 0.1428  # Taxa de recuperação
mu = 0.0025     # Taxa de nascimento/morte

# Condições iniciais, Rio de Janeiro, 2019
S0 = 0.99991    # Fração inicial de suscetíveis
E0 = 0.001      # Fração inicial de expostos
I0 = 0.00008    # Fração inicial de infectados
R0 = 0.0        # Fração inicial de recuperados

# Vetor de tempo. Obs: Aumentar o espaço de tempo neste vetor é importante para visualização a longo prazo
t = np.linspace(0, 200, 1000)

# Solução as equações diferenciais
solution = odeint(model, [S0, E0, I0, R0], t, args=(beta, sigma, gamma, mu))

S, E, I, R = solution.T

plt.plot(t, S, label='Suscetíveis')
plt.plot(t, E, label='Expostos')
plt.plot(t, I, label='Infectados')
plt.plot(t, R, label='Recuperados')
plt.xlabel('Tempo')
plt.ylabel('Fração da população')
plt.title('Modelo SEIR da Epidemiologia com Demografia')
plt.legend()
plt.show()
