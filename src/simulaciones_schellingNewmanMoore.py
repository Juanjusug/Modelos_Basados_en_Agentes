"""
Script de simulación y guardado automático para el modelo matricial de Schelling.

Implementaciones y Características:
1. Clasificación automática del régimen en las 6 dinámicas principales:
   - homofilico_homofilico
   - homofilico_heterofilico
   - homofilico_neutral
   - heterofilico_homofilico
   - heterofilico_heterofilico
   - heterofilico_neutral
   (además de variantes neutrales).
2. Regla de agente aislado: Si un agente no tiene vecinos ocupados (n_a == 0),
   su incomodidad es I_a = tau + 1.0 (agente insatisfecho por defecto).
3. Simulación paralela comparativa de dos tipos de vecindarios (Moore y Newman / Von Neumann)
   con radio 2 bajo las mismas condiciones iniciales.
4. Organización de resultados en subcarpetas según la dinámica clasificada:
   resultados/<tipo_dinamica>/<nombre_corrida>/
   - grafico_incomodidad_comparativo.png
   - estados_moore.png
   - estados_newman.png
   - datos_por_paso_moore.csv
   - datos_por_paso_newman.csv
   - resumen.json
"""

from __future__ import annotations
import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Generación de gráficos sin servidor gráfico (headless)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Importar mesa o utilizar fallback embebido
try:
    import mesa
except ImportError:
    class Agent:
        def __init__(self, model):
            self.model = model
            self.pos = None
            self.random = model.random if hasattr(model, 'random') else random

    class AgentSet(list):
        def shuffle_do(self, method_name: str):
            agents_copy = list(self)
            random.shuffle(agents_copy)
            for agent in agents_copy:
                getattr(agent, method_name)()

    class Model:
        def __init__(self, seed=None):
            self._seed = seed
            self.random = random.Random(seed)
            self.agents = AgentSet()

    class SingleGrid:
        def __init__(self, width: int, height: int, torus: bool = True):
            self.width = width
            self.height = height
            self.torus = torus
            self._grid = {}

        def is_cell_empty(self, pos: tuple[int, int]) -> bool:
            return pos not in self._grid or self._grid[pos] is None

        def place_agent(self, agent: Agent, pos: tuple[int, int]):
            self._grid[pos] = agent
            agent.pos = pos
            if agent not in agent.model.agents:
                agent.model.agents.append(agent)

        def move_agent(self, agent: Agent, new_pos: tuple[int, int]):
            if agent.pos in self._grid and self._grid[agent.pos] == agent:
                del self._grid[agent.pos]
            self._grid[new_pos] = agent
            agent.pos = new_pos

        def get_neighborhood(self, pos: tuple[int, int], moore: bool = True, include_center: bool = False, radius: int = 1) -> list[tuple[int, int]]:
            x, y = pos
            coords = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if not include_center and dx == 0 and dy == 0:
                        continue
                    if not moore and (abs(dx) + abs(dy) > radius):
                        continue
                    nx = x + dx
                    ny = y + dy
                    if self.torus:
                        nx = nx % self.width
                        ny = ny % self.height
                    else:
                        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                            continue
                    coords.append((nx, ny))
            return coords

        def get_neighbors(self, pos: tuple[int, int], moore: bool = True, include_center: bool = False, radius: int = 1) -> list[Agent]:
            neighborhood = self.get_neighborhood(pos, moore=moore, include_center=include_center, radius=radius)
            neighbors = []
            for p in neighborhood:
                if p in self._grid and self._grid[p] is not None:
                    neighbors.append(self._grid[p])
            return neighbors

    class SpaceModule:
        SingleGrid = SingleGrid

    class DataCollector:
        def __init__(self, model_reporters=None):
            self.model_reporters = model_reporters or {}
            self._data = {k: [] for k in self.model_reporters.keys()}

        def collect(self, model):
            for k, reporter in self.model_reporters.items():
                if callable(reporter):
                    val = reporter(model)
                else:
                    val = getattr(model, reporter)
                self._data[k].append(val)

        def get_model_vars_dataframe(self) -> pd.DataFrame:
            return pd.DataFrame(self._data)

    class MesaFallback:
        Agent = Agent
        Model = Model
        space = SpaceModule()
        DataCollector = DataCollector

    mesa = MesaFallback()

CARPETA_RESULTADOS = "resultados"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ========== CLASES DEL MODELO MATRICIAL ==========
class Persona(mesa.Agent):
    """Agente del modelo matricial de Schelling."""

    def __init__(self, model, tipo: int, se_ha_movido: bool = False) -> None:
        super().__init__(model)
        self.tipo = tipo
        self.se_ha_movido = se_ha_movido

    def tolerancia(self):
        if isinstance(self.model.tolerance, dict):
            return self.model.tolerance[self.tipo]
        return self.model.tolerance

    def incomodidad(self, pos, tipo=None):
        if tipo is None:
            tipo = self.tipo

        vecindad = self.model.grid.get_neighbors(
            pos,
            moore=self.model.moore,
            include_center=False,
            radius=self.model.radius,
        )
        n_vecinos = len(vecindad)

        # Regla requerida: Si el agente no tiene vecinos ocupados, I_a = tau + 1.0
        if n_vecinos == 0:
            return float(self.tolerancia() + 1.0)

        i = self.model.tipo_to_idx[tipo]
        total = 0.0
        for vecino in vecindad:
            j = self.model.tipo_to_idx[vecino.tipo]
            total += self.model.H[i, j]

        return float(total / n_vecinos)

    def esta_comodo(self, pos=None):
        if pos is None:
            pos = self.pos
        return self.incomodidad(pos, self.tipo) <= self.tolerancia()

    def move(self):
        self.se_ha_movido = False
        incomodidad_base = self.incomodidad(self.pos, self.tipo)

        if incomodidad_base <= self.tolerancia():
            return

        possible_steps = []
        radio = 0
        max_iteraciones = max(self.model.grid.width, self.model.grid.height) // 2 + 1

        while not possible_steps and radio < max_iteraciones:
            radio += 1
            vecindario = self.model.grid.get_neighborhood(
                self.pos,
                moore=self.model.moore,
                include_center=False,
                radius=radio,
            )
            empty_steps = [step for step in vecindario if self.model.grid.is_cell_empty(step)]
            possible_steps = [
                step for step in empty_steps
                if self.incomodidad(step, self.tipo) < incomodidad_base
            ]

        if possible_steps:
            new_position = self.model.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            self.se_ha_movido = True

    def step(self):
        self.move()


class Modelo_Desplazamiento(mesa.Model):
    """Modelo matricial de Schelling para dos tipos de agentes (1 y -1)."""

    def __init__(self, N1, Nm1, width, height, H, tol, moore=True, radius=1, seed=None, collect_initial=True):
        super().__init__(seed=seed)

        self.N1 = N1
        self.Nm1 = Nm1
        self.num_agents = N1 + Nm1
        self.width = width
        self.height = height
        self.moore = moore
        self.radius = radius

        if self.num_agents > width * height:
            raise ValueError("El número de agentes no puede exceder el número de celdas.")

        self.tipos = [1, -1]
        self.tipo_to_idx = {1: 0, -1: 1}

        self.H = np.array(H, dtype=float)
        if self.H.shape != (2, 2):
            raise ValueError("La matriz H debe tener tamaño 2x2.")

        self.tolerance = tol
        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        self.running = True
        self.in_equilibrium = False
        self.steps_to_equilibrium = 0
        self.moved_count = 0

        self.unsatisfied_count = 0
        self.unsatisfied_by_type = {1: 0, -1: 0}

        self.mean_discomfort = 0.0
        self.mean_discomfort_by_type = {1: 0.0, -1: 0.0}

        self.contact_measure = {
            1: {1: 0.0, -1: 0.0, "vacio": 0.0},
            -1: {1: 0.0, -1: 0.0, "vacio": 0.0},
        }

        self.historial_incomodidad = []

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Steps_to_Equilibrium": "steps_to_equilibrium",
                "In_Equilibrium": "in_equilibrium",
                "Moved_Total": lambda m: m.moved_count,
                "Unsatisfied_Total": lambda m: m.unsatisfied_count,
                "Unsatisfied_1": lambda m: m.unsatisfied_by_type[1],
                "Unsatisfied_-1": lambda m: m.unsatisfied_by_type[-1],
                "Mean_Discomfort": lambda m: m.mean_discomfort,
                "Mean_Discomfort_1": lambda m: m.mean_discomfort_by_type[1],
                "Mean_Discomfort_-1": lambda m: m.mean_discomfort_by_type[-1],
                "C_11": lambda m: m.contact_measure[1][1],
                "C_1m1": lambda m: m.contact_measure[1][-1],
                "C_1vacio": lambda m: m.contact_measure[1]["vacio"],
                "C_m11": lambda m: m.contact_measure[-1][1],
                "C_m1m1": lambda m: m.contact_measure[-1][-1],
                "C_m1vacio": lambda m: m.contact_measure[-1]["vacio"],
            }
        )

        agentes_creados = []
        for _ in range(N1):
            agentes_creados.append(Persona(model=self, tipo=1, se_ha_movido=False))
        for _ in range(Nm1):
            agentes_creados.append(Persona(model=self, tipo=-1, se_ha_movido=False))

        posiciones = [(x, y) for x in range(self.grid.width) for y in range(self.grid.height)]
        self.random.shuffle(posiciones)

        for agent, pos in zip(agentes_creados, posiciones):
            self.grid.place_agent(agent, pos)

        self.update_metrics()
        self.historial_incomodidad.append(self.incomodidad_total())
        if collect_initial:
            self.datacollector.collect(self)

    def incomodidad_total(self) -> float:
        return float(sum(agente.incomodidad(agente.pos) for agente in self.agents))

    def calculate_contact_measure(self):
        acumulado = {
            1: {1: 0.0, -1: 0.0, "vacio": 0.0},
            -1: {1: 0.0, -1: 0.0, "vacio": 0.0},
        }
        n_por_tipo = {1: 0, -1: 0}

        for agent in self.agents:
            n_por_tipo[agent.tipo] += 1
            vecinos = self.grid.get_neighbors(
                agent.pos, moore=self.moore, include_center=False, radius=self.radius
            )
            n_a = len(vecinos)
            capacidad_max = 8 if self.moore else (4 if self.radius == 1 else 12)
            n_a_vacio = max(0, capacidad_max - n_a)

            acumulado[agent.tipo]["vacio"] += n_a_vacio / capacidad_max

            if n_a > 0:
                n_a_j = {1: 0, -1: 0}
                for vecino in vecinos:
                    n_a_j[vecino.tipo] += 1

                acumulado[agent.tipo][1] += n_a_j[1] / n_a
                acumulado[agent.tipo][-1] += n_a_j[-1] / n_a

        self.contact_measure = {
            1: {1: 0.0, -1: 0.0, "vacio": 0.0},
            -1: {1: 0.0, -1: 0.0, "vacio": 0.0},
        }

        for i in self.tipos:
            if n_por_tipo[i] > 0:
                for j in (1, -1, "vacio"):
                    self.contact_measure[i][j] = acumulado[i][j] / n_por_tipo[i]

    def calculate_unsatisfied(self):
        counts = {1: 0, -1: 0}
        for agent in self.agents:
            if agent.incomodidad(agent.pos, agent.tipo) > agent.tolerancia():
                counts[agent.tipo] += 1
        self.unsatisfied_by_type = counts
        self.unsatisfied_count = sum(counts.values())

    def calculate_mean_discomfort(self):
        discomforts_by_type = {1: [], -1: []}
        all_discomforts = []

        for agent in self.agents:
            I = agent.incomodidad(agent.pos, agent.tipo)
            discomforts_by_type[agent.tipo].append(I)
            all_discomforts.append(I)

        self.mean_discomfort = float(np.mean(all_discomforts)) if all_discomforts else 0.0
        for tipo in self.tipos:
            vals = discomforts_by_type[tipo]
            self.mean_discomfort_by_type[tipo] = float(np.mean(vals)) if vals else 0.0

    def calculate_moved_count(self):
        self.moved_count = sum(1 for agent in self.agents if agent.se_ha_movido)

    def update_metrics(self):
        self.calculate_contact_measure()
        self.calculate_unsatisfied()
        self.calculate_mean_discomfort()
        self.calculate_moved_count()

    def check_equilibrium(self) -> bool:
        return self.moved_count == 0

    def step(self):
        if self.in_equilibrium:
            return

        self.agents.shuffle_do("move")
        self.update_metrics()

        if self.check_equilibrium():
            self.in_equilibrium = True
            self.running = False
        else:
            self.steps_to_equilibrium += 1

        self.historial_incomodidad.append(self.incomodidad_total())
        self.datacollector.collect(self)


# ========== CLASIFICACIÓN DE LAS 6 DINÁMICAS ==========
def signo_a_regimen(valor, tol=1e-12):
    if valor > tol:
        return "homofilico"
    if valor < -tol:
        return "heterofilico"
    return "neutral"


def clasificar_regimen(H):
    """
    Clasifica el régimen de cada tipo de agente en las dinámicas principales
    según la matriz H = [[alpha, beta], [delta, gamma]]:
      c1 = beta - alpha
      c_-1 = delta - gamma
    
    Retorna el nombre de la subcarpeta que agrupa la dinámica, por ejemplo:
      - homofilico_homofilico
      - homofilico_heterofilico
      - homofilico_neutral
      - heterofilico_homofilico
      - heterofilico_heterofilico
      - heterofilico_neutral
    """
    H = np.asarray(H, dtype=float)
    alpha, beta = H[0, 0], H[0, 1]
    delta, gamma = H[1, 0], H[1, 1]

    c1 = beta - alpha
    c_m1 = delta - gamma
    A = beta - delta

    reg1 = signo_a_regimen(c1)
    reg2 = signo_a_regimen(c_m1)

    nombre_dinamica = f"{reg1}_{reg2}"

    return {
        "c1": float(c1),
        "c_-1": float(c_m1),
        "regimen_tipo_1": reg1,
        "regimen_tipo_-1": reg2,
        "nombre_dinamica": nombre_dinamica,
        "A": float(A),
        "simetrico": bool(np.isclose(beta, delta)),
    }


def estado_grid(modelo):
    estado = np.zeros((modelo.grid.height, modelo.grid.width), dtype=int)
    for agent in modelo.agents:
        x, y = agent.pos
        estado[y, x] = agent.tipo
    return estado


# ========== FUNCIONES DE GRAFICADO ==========
def graficar_incomodidad_comparativo(hist_moore, hist_newman, clasificacion, ruta_png):
    """Genera el gráfico comparativo de incomodidad total entre vecindario de Moore y Newman/Von Neumann."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(len(hist_moore)), hist_moore, marker="o", color="tab:blue", label="Moore (Radio 1)", linewidth=1.8, markersize=4)
    ax.plot(range(len(hist_newman)), hist_newman, marker="s", color="tab:orange", label="Von Neumann / Newman (Radio 1)", linewidth=1.8, markersize=4)

    ax.set_xlabel("Paso de Simulación", fontsize=10)
    ax.set_ylabel("Incomodidad Total (I_total)", fontsize=10)
    
    titulo = (
        f"Incomodidad Total — Dinámica: {clasificacion['nombre_dinamica']}\n"
        f"Tipo 1: {clasificacion['regimen_tipo_1']} (c1={clasificacion['c1']:.3g}) | "
        f"Tipo -1: {clasificacion['regimen_tipo_-1']} (c-1={clasificacion['c_-1']:.3g})"
    )
    ax.set_title(titulo, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_estado_inicial_final(estado_inicial, estado_final, H, clasificacion, tol, N1, Nm1, tipo_vecindario, ruta_png):
    """Grafica el estado inicial y final de un vecindario con las anotaciones de parámetros."""
    cmap = ListedColormap(["tab:red", "white", "tab:blue"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(estado_inicial, cmap=cmap, norm=norm, origin="lower")
    axes[0].set_title(f"{tipo_vecindario} — Estado inicial")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(estado_final, cmap=cmap, norm=norm, origin="lower")
    axes[1].set_title(f"{tipo_vecindario} — Estado final")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    leyenda = [
        Patch(facecolor="tab:blue", edgecolor="black", label="Tipo 1"),
        Patch(facecolor="tab:red", edgecolor="black", label="Tipo -1"),
        Patch(facecolor="white", edgecolor="black", label="Vacío"),
    ]
    fig.legend(handles=leyenda, loc="center right", bbox_to_anchor=(1.08, 0.5))

    alpha, beta = H[0, 0], H[0, 1]
    delta, gamma = H[1, 0], H[1, 1]

    texto_parametros = (
        f"Vecindario: {tipo_vecindario} (r=1) | H = [[{alpha:.3g}, {beta:.3g}], [{delta:.3g}, {gamma:.3g}]] | tol = {tol}\n"
        f"Dinámica: {clasificacion['nombre_dinamica']} | Tipo 1: {clasificacion['regimen_tipo_1']} (c1={clasificacion['c1']:.3g}) | "
        f"Tipo -1: {clasificacion['regimen_tipo_-1']} (c-1={clasificacion['c_-1']:.3g})"
    )
    fig.suptitle(texto_parametros, fontsize=10, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========== FUNCIÓN PRINCIPAL DE EJECUCIÓN Y CLASIFICACIÓN ==========
def correr_y_guardar_paralelo(
    alpha, beta, delta, gamma,
    N1, Nm1, width, height, tol,
    max_steps=200, seed=None,
    nombre_corrida=None,
    carpeta_base=CARPETA_RESULTADOS,
):
    """
    Simula en paralelo el modelo de Schelling con vecindario de Moore (r=2)
    y Von Neumann / Newman (r=2) bajo la misma semilla inicial.
    
    Clasifica la dinámica de la corrida y guarda todos los archivos en:
      carpeta_base/<nombre_dinamica>/<nombre_corrida>/
    """
    H = np.array([[alpha, beta], [delta, gamma]], dtype=float)
    clasificacion = clasificar_regimen(H)
    nombre_dinamica = clasificacion["nombre_dinamica"]

    if nombre_corrida is None:
        nombre_corrida = datetime.now().strftime("corrida_%Y%m%d_%H%M%S")

    if seed is None:
        seed = np.random.randint(0, 1_000_000)

    # Crear la subcarpeta clasificada según el tipo de dinámica
    carpeta_destino = os.path.join(carpeta_base, nombre_dinamica, nombre_corrida)
    os.makedirs(carpeta_destino, exist_ok=True)

    # 1. Instanciar y correr Modelo Moore (Radio 1)
    modelo_moore = Modelo_Desplazamiento(
        N1=N1, Nm1=Nm1, width=width, height=height, H=H, tol=tol, moore=True, radius=1, seed=seed
    )
    estado_ini_moore = estado_grid(modelo_moore)

    for _ in range(max_steps):
        if modelo_moore.running:
            modelo_moore.step()
        else:
            break
    estado_fin_moore = estado_grid(modelo_moore)
    datos_moore = modelo_moore.datacollector.get_model_vars_dataframe()

    # 2. Instanciar y correr Modelo Von Neumann / Newman (Radio 1)
    modelo_newman = Modelo_Desplazamiento(
        N1=N1, Nm1=Nm1, width=width, height=height, H=H, tol=tol, moore=False, radius=1, seed=seed
    )
    estado_ini_newman = estado_grid(modelo_newman)

    for _ in range(max_steps):
        if modelo_newman.running:
            modelo_newman.step()
        else:
            break
    estado_fin_newman = estado_grid(modelo_newman)
    datos_newman = modelo_newman.datacollector.get_model_vars_dataframe()

    print("============================================================")
    print(f"Corrida: {nombre_corrida}")
    print(f"Dinámica Clasificada: {nombre_dinamica}")
    print("============================================================")
    print(f"H = [[{alpha}, {beta}], [{delta}, {gamma}]]")
    print(f"Régimen tipo  1: {clasificacion['regimen_tipo_1']} (c1 = {clasificacion['c1']:.4f})")
    print(f"Régimen tipo -1: {clasificacion['regimen_tipo_-1']} (c_-1 = {clasificacion['c_-1']:.4f})")
    print(f"Moore (r=2) — Pasos: {len(datos_moore)-1}, Equilibrio: {modelo_moore.in_equilibrium}, Insatisfechos: {modelo_moore.unsatisfied_count}")
    print(f"Newman (r=2) — Pasos: {len(datos_newman)-1}, Equilibrio: {modelo_newman.in_equilibrium}, Insatisfechos: {modelo_newman.unsatisfied_count}")

    # Guardar gráficos
    graficar_incomodidad_comparativo(
        modelo_moore.historial_incomodidad,
        modelo_newman.historial_incomodidad,
        clasificacion,
        os.path.join(carpeta_destino, "grafico_incomodidad_comparativo.png"),
    )

    graficar_estado_inicial_final(
        estado_ini_moore, estado_fin_moore, H, clasificacion, tol, N1, Nm1,
        "Moore", os.path.join(carpeta_destino, "estados_moore.png"),
    )

    graficar_estado_inicial_final(
        estado_ini_newman, estado_fin_newman, H, clasificacion, tol, N1, Nm1,
        "Von Neumann - Newman", os.path.join(carpeta_destino, "estados_newman.png"),
    )

    # Guardar datos CSV
    datos_moore.to_csv(os.path.join(carpeta_destino, "datos_por_paso_moore.csv"), index_label="paso")
    datos_newman.to_csv(os.path.join(carpeta_destino, "datos_por_paso_newman.csv"), index_label="paso")

    # Guardar resumen JSON
    resumen = {
        "nombre_corrida": nombre_corrida,
        "dinamica_clasificada": nombre_dinamica,
        "parametros": {
            "alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma,
            "N1": N1, "Nm1": Nm1, "width": width, "height": height,
            "tol": tol, "max_steps": max_steps, "seed": seed,
        },
        "clasificacion_regimen": clasificacion,
        "resultado_moore": {
            "pasos_ejecutados": int(len(datos_moore) - 1),
            "equilibrio_alcanzado": bool(modelo_moore.in_equilibrium),
            "agentes_incomodos": int(modelo_moore.unsatisfied_count),
            "incomodidad_promedio": float(modelo_moore.mean_discomfort),
            "historial_incomodidad": [float(x) for x in modelo_moore.historial_incomodidad],
        },
        "resultado_newman": {
            "pasos_ejecutados": int(len(datos_newman) - 1),
            "equilibrio_alcanzado": bool(modelo_newman.in_equilibrium),
            "agentes_incomodos": int(modelo_newman.unsatisfied_count),
            "incomodidad_promedio": float(modelo_newman.mean_discomfort),
            "historial_incomodidad": [float(x) for x in modelo_newman.historial_incomodidad],
        },
    }

    with open(os.path.join(carpeta_destino, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"Guardado exitoso en: {os.path.abspath(carpeta_destino)}\n")
    return resumen


if __name__ == "__main__":
    # Ejemplo de ejecución demostrativa para las 6 dinámicas
    dinamicas_ejemplo = [
        {"nombre": "Caso_Homofilico_Homofilico1", "H": [0, 1, 1, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Homofilico2", "H": [0, 1, 0.9, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Homofilico3", "H": [0, 1, 0.8, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Homofilico4", "H": [0, 1, 0.7, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Homofilico5", "H": [0, 1, 0.6, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Neutral", "H": [0, 1, 0.5, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Heterofilico1", "H": [0, 1, 0.4, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Heterofilico2", "H": [0, 1, 0.3, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Heterofilico3", "H": [0, 1, 0.2, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Heterofilico4", "H": [0, 1, 0.1, 0.5], "tol": 0.1},
        {"nombre": "Caso_Homofilico_Heterofilico5", "H": [0, 1, 0.0, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Homofilico1", "H": [0, -1, 1, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Homofilico2", "H": [0, -1, 0.9, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Homofilico3", "H": [0, -1, 0.8, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Homofilico4", "H": [0, -1, 0.7, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Homofilico5", "H": [0, -1, 0.6, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Neutral", "H": [0, -1, 0.5, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Heterofilico1", "H": [0, -1, 0.4, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Heterofilico2", "H": [0, -1, 0.3, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Heterofilico3", "H": [0, -1, 0.2, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Heterofilico4", "H": [0, -1, 0.1, 0.5], "tol": 0.1},
        {"nombre": "Caso_Heterofilico_Heterofilico5", "H": [0, -1, 0.0, 0.5], "tol": 0.1},
    ]

    for d in dinamicas_ejemplo:
        alpha, beta, delta, gamma = d["H"]
        correr_y_guardar_paralelo(
            alpha=alpha, beta=beta, delta=delta, gamma=gamma,
            N1=750, Nm1=750, width=40, height=40, tol=d["tol"],
            max_steps=50, seed=123,
            nombre_corrida=d["nombre"]
        )
