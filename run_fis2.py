import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

raw_ambiental = pd.read_csv("python_move/ambiental_1000m_normal.csv")
raw_socioeconomico = pd.read_csv("python_move/socioeconomico_1000m_normal.csv")
raw_estrategico = pd.read_csv("python_move/estrategico_1000m_normal.csv")

consolidated = pd.concat([raw_ambiental, raw_socioeconomico["2"], raw_estrategico["2"]], axis=1)
consolidated.columns = ["X", "Y", "ambiental", "socioeconomico", "estrategico"]
consolidated["saida"] = 0


ambiental = ctrl.Antecedent(np.arange(0, 11, 1), "ambiental")
socioeconomico = ctrl.Antecedent(np.arange(0, 11, 1), "socioeconomico")
estrategico = ctrl.Antecedent(np.arange(0, 11, 1), "estrategico")
saida = ctrl.Consequent(np.arange(0, 11, 1), "saida")

## Destaque.R - Refactor to py
# Final 1
ambiental["baixo"] = fuzz.trapmf(ambiental.universe, [0, 0, 2, 4])
ambiental["medio"] = fuzz.trapmf(ambiental.universe, [2, 4, 6, 7])
ambiental["alto"] = fuzz.trapmf(ambiental.universe, [6, 7, 10, 10])

socioeconomico["baixo"] = fuzz.trapmf(socioeconomico.universe, [0, 0, 2, 5])
socioeconomico["medio"] = fuzz.trapmf(socioeconomico.universe, [2, 5, 6, 8])
socioeconomico["alto"] = fuzz.trapmf(socioeconomico.universe, [6, 8, 10, 10])

estrategico["baixo"] = fuzz.trapmf(estrategico.universe, [0, 0, 3, 5])
estrategico["medio"] = fuzz.trapmf(estrategico.universe, [3, 5, 7, 8])
estrategico["alto"] = fuzz.trapmf(estrategico.universe, [7, 8, 10, 10])


saida["MBx"] = fuzz.trimf(saida.universe, [0, 0, 1.5])
saida["BX"] = fuzz.trimf(saida.universe, [0, 1.5, 3.5])
saida["BxM"] = fuzz.trimf(saida.universe, [1.5, 3.5, 5])
saida["Med"] = fuzz.trimf(saida.universe, [3.5, 5, 6.5])
saida["MA"] = fuzz.trimf(saida.universe, [5, 6.5, 8.5])
saida["AL"] = fuzz.trimf(saida.universe, [6.5, 8.5, 10])
saida["MAL"] = fuzz.trimf(saida.universe, [8.5, 10, 10])

# regras
regra1 = ctrl.Rule(ambiental['baixo'] & socioeconomico['baixo'] & estrategico['alto'], saida["MBx"])
regra2 = ctrl.Rule(ambiental['baixo'] & socioeconomico['baixo'] & estrategico['medio'], saida["BX"])
regra3 = ctrl.Rule(ambiental['baixo'] & socioeconomico['baixo'] & estrategico['baixo'], saida["BxM"])
regra4 = ctrl.Rule(ambiental['baixo'] & socioeconomico['medio'] & estrategico['alto'], saida["BX"])
regra5 = ctrl.Rule(ambiental['baixo'] & socioeconomico['medio'] & estrategico['medio'], saida["BxM"])
regra6 = ctrl.Rule(ambiental['baixo'] & socioeconomico['medio'] & estrategico['baixo'], saida["Med"])
regra7 = ctrl.Rule(ambiental['baixo'] & socioeconomico['alto'] & estrategico['alto'], saida["BxM"])
regra8 = ctrl.Rule(ambiental['baixo'] & socioeconomico['alto'] & estrategico['medio'], saida["Med"])
regra9 = ctrl.Rule(ambiental['baixo'] & socioeconomico['alto'] & estrategico['baixo'], saida["MA"])
regra10 = ctrl.Rule(ambiental['medio'] & socioeconomico['baixo'] & estrategico['alto'], saida["BX"])
regra11 = ctrl.Rule(ambiental['medio'] & socioeconomico['baixo'] & estrategico['medio'], saida["BxM"])
regra12 = ctrl.Rule(ambiental['medio'] & socioeconomico['baixo'] & estrategico['baixo'], saida["Med"])
regra13 = ctrl.Rule(ambiental['medio'] & socioeconomico['medio'] & estrategico['alto'], saida["BxM"])
regra14 = ctrl.Rule(ambiental['medio'] & socioeconomico['medio'] & estrategico['medio'], saida["Med"])
regra15 = ctrl.Rule(ambiental['medio'] & socioeconomico['medio'] & estrategico['baixo'], saida["MA"])
regra16 = ctrl.Rule(ambiental['medio'] & socioeconomico['alto'] & estrategico['alto'], saida["Med"])
regra17 = ctrl.Rule(ambiental['medio'] & socioeconomico['alto'] & estrategico['medio'], saida["MA"])
regra18 = ctrl.Rule(ambiental['medio'] & socioeconomico['alto'] & estrategico['baixo'], saida["AL"])
regra19 = ctrl.Rule(ambiental['alto'] & socioeconomico['baixo'] & estrategico['alto'], saida["Med"])
regra20 = ctrl.Rule(ambiental['alto'] & socioeconomico['baixo'] & estrategico['medio'], saida["Med"])
regra21 = ctrl.Rule(ambiental['alto'] & socioeconomico['baixo'] & estrategico['baixo'], saida["MA"])
regra22 = ctrl.Rule(ambiental['alto'] & socioeconomico['medio'] & estrategico['alto'], saida["Med"])
regra23 = ctrl.Rule(ambiental['alto'] & socioeconomico['medio'] & estrategico['medio'], saida["MA"])
regra24 = ctrl.Rule(ambiental['alto'] & socioeconomico['medio'] & estrategico['baixo'], saida["AL"])
regra25 = ctrl.Rule(ambiental['alto'] & socioeconomico['alto'] & estrategico['alto'], saida["MA"])
regra26 = ctrl.Rule(ambiental['alto'] & socioeconomico['alto'] & estrategico['medio'], saida["AL"])
regra27 = ctrl.Rule(ambiental['alto'] & socioeconomico['alto'] & estrategico['baixo'], saida["MAL"])
destaque_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8, regra9 ,regra10, regra11, regra12, regra13, regra14, regra15, regra16, regra17, regra18, regra19, regra20, regra21, regra22, regra23, regra24, regra25, regra26, regra27])

destaque = ctrl.ControlSystemSimulation(destaque_ctrl)


for index, row in consolidated.iterrows():
    try:
        destaque.input["ambiental"] = row["ambiental"]
        destaque.input["socioeconomico"] = row["socioeconomico"]
        destaque.input["estrategico"] = row["estrategico"]
        destaque.compute()
        consolidated.loc[index, "saida"] = destaque.output["saida"]
    except Exception as e:
        print(e, "error", index)
        print(row)
        consolidated.loc[index, "saida"] = 0


consolidated.to_csv("python_move/consolidated2.csv", index=False)
