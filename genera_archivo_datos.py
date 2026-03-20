import pandas as pd
import random

n = 3000

data = []
for i in range(n):
    edad = random.randint(18, 75)
    antiguedad = random.randint(1, 60)
    gasto = random.randint(8000, 120000)
    soporte = random.randint(0, 8)
    tipo_contrato = random.choice(["mensual", "anual", "bienal"])
    region = random.choice(["norte", "centro", "sur"])
    
    # lógica realista de churn
    churn_prob = 0.1
    if antiguedad < 6:
        churn_prob += 0.25
    if soporte > 4:
        churn_prob += 0.3
    if gasto < 20000:
        churn_prob += 0.2
    if tipo_contrato == "mensual":
        churn_prob += 0.15
    
    churn = 1 if random.random() < churn_prob else 0
    
    data.append([
        edad, antiguedad, gasto, soporte, tipo_contrato, region, churn
    ])

df = pd.DataFrame(data, columns=[
    "edad", "antiguedad_meses", "gasto_mensual",
    "soporte_llamadas", "tipo_contrato", "region", "churn"
])

# Guardar archivo CSV
# df.to_csv("dataset_churn_empresa_realista.csv", index=False)
df.to_csv("datas.csv", index=False)

print("Archivo generado correctamente")