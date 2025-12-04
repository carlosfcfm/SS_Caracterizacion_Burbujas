import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
# Ajusta esta ruta a la carpeta donde están tus 208 archivos CSV
RUTA_CARPETA = "C:/Users/LamdaZero/Desktop/Datos sensor sensirion"  # Cambia esto a tu ruta real
SEPARADOR_BURBUJAS = "--- DATOS DE BURBUJAS INDIVIDUALES ---"

# ============================================================================
# FUNCIONES
# ============================================================================

def leer_csv_con_dos_partes(filepath):
    """
    Lee un CSV que tiene dos partes separadas por el texto separador.
    Retorna: (df_continuo, df_burbujas_individuales)
    """
    try:
        with open(filepath, 'r') as f:
            contenido = f.read()
        
        if SEPARADOR_BURBUJAS not in contenido:
            print(f"⚠️ Advertencia: {filepath.name} no tiene datos de burbujas individuales")
            return None, None
        
        partes = contenido.split(SEPARADOR_BURBUJAS)
        
        # Parte 1: Datos continuos
        from io import StringIO
        df_continuo = pd.read_csv(StringIO(partes[0]))
        
        # Parte 2: Datos de burbujas individuales
        df_burbujas = pd.read_csv(StringIO(partes[1]))
        
        return df_continuo, df_burbujas
    
    except Exception as e:
        print(f"❌ Error leyendo {filepath}: {e}")
        return None, None


def validar_contador_burbujas(df_continuo):
    """
    Verifica si el contador de burbujas empieza en 0.
    Si no, retorna el índice donde sí empieza en 0 o salta correctamente.
    """
    if df_continuo is None or 'burb_count' not in df_continuo.columns:
        return 0
    
    primer_valor = df_continuo['burb_count'].iloc[0]
    
    if primer_valor == 0:
        return 0
    
    # Buscar donde el contador resetea o empieza de forma válida
    for i in range(1, len(df_continuo)):
        if df_continuo['burb_count'].iloc[i] == 0:
            return i
        # Si encontramos un salto (ej: de 2 a 3), usamos desde ahí
        if i > 0 and df_continuo['burb_count'].iloc[i] == df_continuo['burb_count'].iloc[i-1] + 1:
            if df_continuo['burb_count'].iloc[i-1] != 0:
                return i-1
    
    # Si nunca empieza en 0, usar desde el segundo valor
    return 1


def obtener_datos_finales(df_continuo, indice_inicio):
    """
    Obtiene los datos de la última fila válida del continuo (después del índice de inicio).
    """
    if df_continuo is None or len(df_continuo) == 0:
        return None
    
    df_valido = df_continuo.iloc[indice_inicio:]
    
    if len(df_valido) == 0:
        return None
    
    ultima_fila = df_valido.iloc[-1]
    
    return {
        'volumen_total': ultima_fila.get('volume_total_L', np.nan),
        'error_volumen_total': ultima_fila.get('volume_total_error_L', np.nan),
        'num_burbujas': ultima_fila.get('burb_count', np.nan),
        'flow_filtrado': ultima_fila.get('flow_filtrado_slm', np.nan),
        'temperatura': ultima_fila.get('temp_C', np.nan),
        'presion': ultima_fila.get('press_hPa', np.nan)
    }


# ============================================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================================

print("="*70)
print("ANÁLISIS DE BURBUJAS - 208 ARCHIVOS CSV")
print("="*70)

# Listar todos los archivos CSV
ruta = Path(RUTA_CARPETA)
archivos_csv = list(ruta.glob("*.csv"))

if len(archivos_csv) == 0:
    print(f"❌ No se encontraron archivos CSV en {RUTA_CARPETA}")
    print("Por favor, verifica la ruta y asegúrate de que los archivos existan.")
    exit()

print(f"\n📁 Se encontraron {len(archivos_csv)} archivos CSV")

# Listas para almacenar todos los datos
todos_datos_individuales = []
todos_errores_totales = []
todos_errores_individuales = []

archivos_procesados = 0
archivos_con_error = 0

# Procesar cada archivo
for i, archivo in enumerate(archivos_csv, 1):
    print(f"\nProcesando archivo {i}/{len(archivos_csv)}: {archivo.name}")
    
    df_continuo, df_burbujas = leer_csv_con_dos_partes(archivo)
    
    if df_continuo is None or df_burbujas is None:
        archivos_con_error += 1
        continue
    
    # Validar contador de burbujas
    indice_inicio = validar_contador_burbujas(df_continuo)
    if indice_inicio > 0:
        print(f"  ⚠️ Contador no empieza en 0. Usando datos desde índice {indice_inicio}")
    
    # Obtener datos finales del continuo
    datos_finales = obtener_datos_finales(df_continuo, indice_inicio)
    
    if datos_finales and not np.isnan(datos_finales['error_volumen_total']):
        todos_errores_totales.append(datos_finales['error_volumen_total'])
    
    # Procesar burbujas individuales
    if len(df_burbujas) > 0:
        # DIAGNÓSTICO: Verificar si hay valores anómalos
        if 'vol_por_burbuja_L' in df_burbujas.columns:
            max_vol = df_burbujas['vol_por_burbuja_L'].max()
            if max_vol > 0.010:  # Umbral de alerta: burbujas > 10 mL
                print(f"  ⚠️ ALERTA: Volumen máximo por burbuja = {max_vol:.6f} L")
                print(f"     Archivo: {archivo.name}")
                print(f"     Número de burbujas con vol > 0.010 L: {(df_burbujas['vol_por_burbuja_L'] > 0.010).sum()}")
        
        # Agregar variables ambientales promedio del continuo (o usar las de cada burbuja si están)
        if 'temp_C' not in df_burbujas.columns and datos_finales:
            df_burbujas['temp_C'] = datos_finales['temperatura']
        if 'press_hPa' not in df_burbujas.columns and datos_finales:
            df_burbujas['press_hPa'] = datos_finales['presion']
        if 'flow_filtrado_slm' not in df_burbujas.columns and datos_finales:
            df_burbujas['flow_filtrado_slm'] = datos_finales['flow_filtrado']
        
        # Agregar columna de archivo origen
        df_burbujas['archivo'] = archivo.stem
        
        todos_datos_individuales.append(df_burbujas)
        
        # Calcular errores individuales
        if 'vol_por_burbuja_error_L' in df_burbujas.columns:
            errores = df_burbujas['vol_por_burbuja_error_L'].dropna()
            todos_errores_individuales.extend(errores.tolist())
    
    archivos_procesados += 1

print("\n" + "="*70)
print(f"✅ Archivos procesados exitosamente: {archivos_procesados}")
print(f"❌ Archivos con errores: {archivos_con_error}")
print("="*70)

# ============================================================================
# CONSOLIDAR DATOS
# ============================================================================

if len(todos_datos_individuales) == 0:
    print("\n❌ No se encontraron datos de burbujas individuales para analizar.")
    exit()

df_todas_burbujas = pd.concat(todos_datos_individuales, ignore_index=True)

print(f"\n📊 Total de burbujas recopiladas: {len(df_todas_burbujas)}")
print(f"📊 Total de archivos con datos válidos: {len(todos_datos_individuales)}")

# DIAGNÓSTICO DETALLADO DE VALORES ANÓMALOS
print("\n" + "="*70)
print("DIAGNÓSTICO DE VALORES ANÓMALOS")
print("="*70)

if 'vol_por_burbuja_L' in df_todas_burbujas.columns:
    print(f"\n📈 Estadísticas de volumen por burbuja:")
    print(f"   Mínimo: {df_todas_burbujas['vol_por_burbuja_L'].min():.6f} L")
    print(f"   Máximo: {df_todas_burbujas['vol_por_burbuja_L'].max():.6f} L")
    print(f"   Media: {df_todas_burbujas['vol_por_burbuja_L'].mean():.6f} L")
    print(f"   Mediana: {df_todas_burbujas['vol_por_burbuja_L'].median():.6f} L")
    
    # Identificar valores sospechosos (> 0.010 L o 10 mL)
    burbujas_sospechosas = df_todas_burbujas[df_todas_burbujas['vol_por_burbuja_L'] > 0.010]
    
    if len(burbujas_sospechosas) > 0:
        print(f"\n⚠️ Se encontraron {len(burbujas_sospechosas)} burbujas con volumen > 0.010 L:")
        print("\nArchivos afectados:")
        archivos_con_problemas = burbujas_sospechosas.groupby('archivo')['vol_por_burbuja_L'].agg(['count', 'max'])
        print(archivos_con_problemas.to_string())
        
        print("\n🔍 Primeros 10 valores anómalos:")
        print(burbujas_sospechosas[['archivo', 'vol_por_burbuja_L', 'bubble_time_s']].head(10).to_string(index=False))
    else:
        print("\n✅ No se encontraron valores anómalos (todos los volúmenes < 0.010 L)")
    
    # Sugerencia de filtro
    percentil_99 = df_todas_burbujas['vol_por_burbuja_L'].quantile(0.99)
    print(f"\n💡 Percentil 99 del volumen: {percentil_99:.6f} L")
    print(f"   (Considera filtrar valores por encima de este umbral)")


# ============================================================================
# CÁLCULO DE ERRORES PROMEDIO
# ============================================================================

print("\n" + "="*70)
print("ERRORES PROMEDIO")
print("="*70)

if len(todos_errores_totales) > 0:
    error_promedio_total = np.mean(todos_errores_totales)
    std_error_total = np.std(todos_errores_totales)
    print(f"\n📈 Error promedio en volumen total (última medición):")
    print(f"   Media: {error_promedio_total:.6f} L")
    print(f"   Desviación estándar: {std_error_total:.6f} L")
else:
    print("\n⚠️ No se encontraron datos de error de volumen total")

if len(todos_errores_individuales) > 0:
    error_promedio_individual = np.mean(todos_errores_individuales)
    std_error_individual = np.std(todos_errores_individuales)
    print(f"\n📈 Error promedio en volumen individual por burbuja:")
    print(f"   Media: {error_promedio_individual:.6f} L")
    print(f"   Desviación estándar: {std_error_individual:.6f} L")
else:
    print("\n⚠️ No se encontraron datos de error de volumen individual")

# ============================================================================
# ANÁLISIS ESTADÍSTICO
# ============================================================================

print("\n" + "="*70)
print("ANÁLISIS ESTADÍSTICO")
print("="*70)

# Verificar que tengamos las columnas necesarias
columnas_necesarias = ['vol_por_burbuja_L', 'flow_filtrado_slm', 'temp_C', 'press_hPa']
columnas_faltantes = [col for col in columnas_necesarias if col not in df_todas_burbujas.columns]

if columnas_faltantes:
    print(f"\n❌ Faltan columnas necesarias para análisis: {columnas_faltantes}")
    exit()

# Limpiar datos (eliminar NaN)
df_limpio = df_todas_burbujas[columnas_necesarias].dropna()
print(f"\n✅ Burbujas válidas para análisis: {len(df_limpio)} de {len(df_todas_burbujas)}")

# Estadísticas descriptivas
print("\n" + "-"*70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("-"*70)
print(df_limpio.describe())

# Categorizar variables continuas en grupos para ANOVA
try:
    df_limpio['flow_grupo'] = pd.qcut(df_limpio['flow_filtrado_slm'], q=4, duplicates='drop')
except Exception as e:
    print(f"⚠️ Advertencia al categorizar flow: {e}")
    df_limpio['flow_grupo'] = pd.cut(df_limpio['flow_filtrado_slm'], bins=4)

try:
    df_limpio['temp_grupo'] = pd.qcut(df_limpio['temp_C'], q=4, duplicates='drop')
except Exception as e:
    print(f"⚠️ Advertencia al categorizar temperatura: {e}")
    df_limpio['temp_grupo'] = pd.cut(df_limpio['temp_C'], bins=4)

try:
    df_limpio['press_grupo'] = pd.qcut(df_limpio['press_hPa'], q=4, duplicates='drop')
except Exception as e:
    print(f"⚠️ Advertencia al categorizar presión: {e}")
    df_limpio['press_grupo'] = pd.cut(df_limpio['press_hPa'], bins=4)

# ============================================================================
# ANOVA MULTIFACTORIAL
# ============================================================================

print("\n" + "-"*70)
print("ANOVA MULTIFACTORIAL: Efectos combinados de Flow, Temperatura y Presión")
print("-"*70)

# Fórmula para ANOVA con efectos principales e interacciones (puedes ajustar)
formula = 'vol_por_burbuja_L ~ C(flow_grupo) + C(temp_grupo) + C(press_grupo) + C(flow_grupo):C(temp_grupo) + C(flow_grupo):C(press_grupo) + C(temp_grupo):C(press_grupo)'
model_anova = ols(formula, data=df_limpio).fit()
anova_table = anova_lm(model_anova, typ=2)
print(anova_table)

# Interpretación
print("\n💡 Interpretación ANOVA:")
for index, row in anova_table.iterrows():
    if row['PR(>F)'] < 0.05:
        print(f"   ✅ {index}: Efecto SIGNIFICATIVO (p = {row['PR(>F)']:.6f})")
    else:
        print(f"   ❌ {index}: NO significativo (p = {row['PR(>F)']:.6f})")

# ============================================================================
# POST-HOC PARA ANOVA (SI SIGNIFICATIVO)
# ============================================================================

print("\n" + "-"*70)
print("POST-HOC (Tukey HSD) PARA EFECTOS SIGNIFICATIVOS")
print("-"*70)

# Post-hoc para cada factor principal si su p < 0.05 en ANOVA
for factor in ['flow_grupo', 'temp_grupo', 'press_grupo']:
    if anova_table.loc[f'C({factor})', 'PR(>F)'] < 0.05:
        print(f"\n📊 Tukey HSD para {factor}:")
        tukey = pairwise_tukeyhsd(df_limpio['vol_por_burbuja_L'], df_limpio[factor])
        print(tukey)
    else:
        print(f"\n⚠️ No se realiza Tukey para {factor} (no significativo en ANOVA)")

# ============================================================================
# CORRELACIONES: HEATMAP
# ============================================================================

print("\n" + "-"*70)
print("HEATMAP DE CORRELACIONES")
print("-"*70)

corr_matrix = df_limpio[columnas_necesarias].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Matriz de Correlaciones de Pearson')
plt.savefig('heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
print("\n✅ Heatmap guardada como: heatmap_correlaciones.png")
plt.show()

# ============================================================================
# REGRESIÓN LINEAL MÚLTIPLE
# ============================================================================

print("\n" + "-"*70)
print("REGRESIÓN LINEAL MÚLTIPLE (Efectos combinados)")
print("-"*70)

X = sm.add_constant(df_limpio[['flow_filtrado_slm', 'temp_C', 'press_hPa']])
y = df_limpio['vol_por_burbuja_L']
model_reg = sm.OLS(y, X).fit()
print(model_reg.summary())

print("\n💡 Interpretación:")
for var, p in zip(X.columns[1:], model_reg.pvalues[1:]):
    sig = "SIGNIFICATIVO" if p < 0.05 else "NO significativo"
    print(f"   {var}: p-value = {p:.6f} ({sig})")
print(f"   R² ajustado: {model_reg.rsquared_adj:.4f} (explica {model_reg.rsquared_adj*100:.1f}% de la varianza)")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)
print(f"\n✅ Archivos procesados: {archivos_procesados}/{len(archivos_csv)}")
print(f"✅ Total de burbujas analizadas: {len(df_limpio)}")
print(f"\n📈 Errores promedio:")
if len(todos_errores_totales) > 0:
    print(f"   • Volumen total: {error_promedio_total:.6f} ± {std_error_total:.6f} L")
if len(todos_errores_individuales) > 0:
    print(f"   • Volumen individual: {error_promedio_individual:.6f} ± {std_error_individual:.6f} L")

print("\n💡 Recomendación:")
print("   Revisa los valores de p-value en el análisis ANOVA multifactorial, post-hoc, correlaciones y regresión múltiple")
print("   para determinar qué variables y sus interacciones tienen mayor efecto sobre el volumen de burbujas.")
print("   Valores de p < 0.05 indican efectos estadísticamente significativos.")

# ============================================================================
# GRÁFICAS CON REGRESIONES LINEALES
# ============================================================================

print("\n" + "="*70)
print("GENERANDO GRÁFICAS")
print("="*70)

# Crear figura con las 3 variables vs volumen
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Regresiones Lineales: Efectos sobre Volumen por Burbuja Individual', 
             fontsize=16, fontweight='bold')

# 1. Flow Filtrado vs Volumen
slope_flow, intercept_flow, r_value_flow, p_value_flow, std_err_flow = stats.linregress(
    df_limpio['flow_filtrado_slm'], df_limpio['vol_por_burbuja_L'])

axes[0].scatter(df_limpio['flow_filtrado_slm'], df_limpio['vol_por_burbuja_L'], 
                alpha=0.4, s=15, color='coral', label='Datos')
x_flow = np.linspace(df_limpio['flow_filtrado_slm'].min(), 
                     df_limpio['flow_filtrado_slm'].max(), 100)
axes[0].plot(x_flow, slope_flow * x_flow + intercept_flow, 
             'b-', linewidth=2.5, label='Regresión lineal')
axes[0].set_xlabel('Flow Filtrado (SLM)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Volumen por Burbuja (L)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Flow vs Volumen\nR² = {r_value_flow**2:.4f}, p = {p_value_flow:.2e}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='best')
# Agregar ecuación
axes[0].text(0.05, 0.95, f'y = {slope_flow:.6f}x + {intercept_flow:.6f}', 
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Temperatura vs Volumen
slope_temp, intercept_temp, r_value_temp, p_value_temp, std_err_temp = stats.linregress(
    df_limpio['temp_C'], df_limpio['vol_por_burbuja_L'])

axes[1].scatter(df_limpio['temp_C'], df_limpio['vol_por_burbuja_L'], 
                alpha=0.4, s=15, color='orange', label='Datos')
x_temp = np.linspace(df_limpio['temp_C'].min(), 
                     df_limpio['temp_C'].max(), 100)
axes[1].plot(x_temp, slope_temp * x_temp + intercept_temp, 
             'g-', linewidth=2.5, label='Regresión lineal')
axes[1].set_xlabel('Temperatura (°C)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Volumen por Burbuja (L)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Temperatura vs Volumen\nR² = {r_value_temp**2:.4f}, p = {p_value_temp:.2e}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='best')
# Agregar ecuación
axes[1].text(0.05, 0.95, f'y = {slope_temp:.6f}x + {intercept_temp:.6f}', 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Presión vs Volumen
slope_press, intercept_press, r_value_press, p_value_press, std_err_press = stats.linregress(
    df_limpio['press_hPa'], df_limpio['vol_por_burbuja_L'])

axes[2].scatter(df_limpio['press_hPa'], df_limpio['vol_por_burbuja_L'], 
                alpha=0.4, s=15, color='steelblue', label='Datos')
x_press = np.linspace(df_limpio['press_hPa'].min(), 
                      df_limpio['press_hPa'].max(), 100)
axes[2].plot(x_press, slope_press * x_press + intercept_press, 
             'r-', linewidth=2.5, label='Regresión lineal')
axes[2].set_xlabel('Presión (hPa)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Volumen por Burbuja (L)', fontsize=12, fontweight='bold')
axes[2].set_title(f'Presión vs Volumen\nR² = {r_value_press**2:.4f}, p = {p_value_press:.2e}', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='best')
# Agregar ecuación
axes[2].text(0.05, 0.95, f'y = {slope_press:.6f}x + {intercept_press:.6f}', 
             transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('regresiones_variables_vs_volumen.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfica guardada como: regresiones_variables_vs_volumen.png")
print("   Esta gráfica muestra las regresiones lineales de las 3 variables")
print("   contra el volumen por burbuja INDIVIDUAL")

plt.show()

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)