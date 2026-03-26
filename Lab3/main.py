import tkinter as tk
import random
import sqlite3
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

CELL_SIZE = 50
TUNNEL_WIDTH = 12
TUNNEL_HEIGHT = 6
VEHICLE_COLORS = ["red", "blue", "green"]
EXHAUST_ZONE = [(0, 2), (0, 3), (11, 2), (11, 3)]
DB_NAME = "tunnel.db"


class Vehicle:
    def __init__(self, name, x, y, color, vehicle_type="car"):
        self.name = name
        self.x = x
        self.y = y
        self.color = color
        self.vehicle_type = vehicle_type
        self.status = "moving"
        self.emission_rate = 1.0 if vehicle_type == "car" else 2.5 if vehicle_type == "truck" else 0.5


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ventilation_rules (
        id INTEGER PRIMARY KEY,
        condition TEXT,
        action TEXT,
        priority INTEGER
    )
    """)
    c.execute("DELETE FROM ventilation_rules")
    rules = [
        ("air_quality < 40 and fans_speed < 80", "fans_speed = min(100, fans_speed + 25)", 1),
        ("vehicle_count > 10 and fans_speed < 70", "fans_speed = min(100, fans_speed + 20)", 2),
        ("air_quality < 60 and vehicle_count > 5 and fans_speed < 60", "fans_speed = min(100, fans_speed + 15)", 3),
        ("air_quality < 70 and fans_speed < 50", "fans_speed = min(100, fans_speed + 10)", 4),
        ("vehicle_count < 3 and air_quality > 70", "fans_speed = max(0, fans_speed - 35)", 10),
        ("vehicle_count == 0", "fans_speed = max(0, fans_speed - 50)", 11),
        ("air_quality > 80 and fans_speed > 20", "fans_speed = max(20, fans_speed - 25)", 9),
        ("air_quality > 75 and vehicle_count < 5", "fans_speed = max(15, fans_speed - 30)", 8),
        ("air_quality > 73 and fans_speed > 30", "fans_speed = max(30, fans_speed - 15)", 7),
    ]
    c.executemany("INSERT INTO ventilation_rules(condition, action, priority) VALUES (?, ?, ?)", rules)
    conn.commit()
    conn.close()


class TunnelVentilationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система управления вентиляцией туннеля")
        self.canvas = tk.Canvas(root, width=CELL_SIZE * TUNNEL_WIDTH, height=CELL_SIZE * TUNNEL_HEIGHT)
        self.canvas.pack(side="left")

        control_frame = tk.Frame(root)
        control_frame.pack(side="right", fill="y")

        self.log_text = tk.Text(control_frame, width=50, height=15)
        self.log_text.pack(fill="both", expand=True)

        self.info_label = tk.Label(control_frame, text="", font=("Arial", 10))
        self.info_label.pack(pady=5)

        init_db()
        self.vehicles = []
        self.create_initial_vehicles()

        self.air_quality = 45
        self.fans_speed = 35
        self.vehicle_count = len(self.vehicles)
        self.step_count = 0

        self.step_button = tk.Button(control_frame, text="Следующий шаг", command=self.step)
        self.step_button.pack(pady=5)
        self.reset_button = tk.Button(control_frame, text="Сброс", command=self.reset)
        self.reset_button.pack(pady=5)

        self.setup_fuzzy_logic()
        self.draw_tunnel()
        self.update_info_display()
        self.log_text.insert(tk.END, "Система управления вентиляцией туннеля запущена\n")

    def create_initial_vehicles(self):
        vehicle_data = [
            (0, 1, "car", "blue"),
            (0, 4, "truck", "red"),
            (2, 2, "motorcycle", "green"),
            (4, 3, "car", "blue"),
            (6, 1, "truck", "red"),
            (8, 4, "motorcycle", "green")
        ]
        for i, (x, y, vehicle_type, color) in enumerate(vehicle_data):
            self.vehicles.append(Vehicle(f"V{i + 1}", x, y, color, vehicle_type))

    def setup_fuzzy_logic(self):
        self.vehicle_input = ctrl.Antecedent(np.arange(0, 31, 1), 'vehicle_count')
        self.vehicle_input['low'] = fuzz.trimf(self.vehicle_input.universe, [0, 0, 5])
        self.vehicle_input['medium'] = fuzz.trimf(self.vehicle_input.universe, [3, 8, 15])
        self.vehicle_input['high'] = fuzz.trimf(self.vehicle_input.universe, [10, 20, 30])

        self.quality_input = ctrl.Antecedent(np.arange(0, 101, 1), 'air_quality')
        self.quality_input['poor'] = fuzz.trimf(self.quality_input.universe, [0, 0, 40])
        self.quality_input['moderate'] = fuzz.trimf(self.quality_input.universe, [30, 50, 70])
        self.quality_input['good'] = fuzz.trimf(self.quality_input.universe, [60, 100, 100])

        self.fans_output = ctrl.Consequent(np.arange(0, 101, 1), 'fans_speed')
        self.fans_output['low'] = fuzz.trimf(self.fans_output.universe, [0, 0, 30])
        self.fans_output['medium'] = fuzz.trimf(self.fans_output.universe, [20, 50, 80])
        self.fans_output['high'] = fuzz.trimf(self.fans_output.universe, [70, 100, 100])

        rule1 = ctrl.Rule(self.vehicle_input['low'] & self.quality_input['good'], self.fans_output['low'])
        rule2 = ctrl.Rule(self.vehicle_input['medium'] | self.quality_input['moderate'], self.fans_output['medium'])
        rule3 = ctrl.Rule(self.vehicle_input['high'] | self.quality_input['poor'], self.fans_output['high'])

        self.ventilation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.ventilation_sim = ctrl.ControlSystemSimulation(self.ventilation_ctrl)

    def draw_tunnel(self):
        self.canvas.delete("all")

        # Рисуем туннель
        for i in range(TUNNEL_WIDTH):
            for j in range(TUNNEL_HEIGHT):
                color = "lightgray" if j in [0, TUNNEL_HEIGHT - 1] else "white"
                if (i, j) in EXHAUST_ZONE:
                    color = "brown"
                self.canvas.create_rectangle(i * CELL_SIZE, j * CELL_SIZE,
                                             (i + 1) * CELL_SIZE, (j + 1) * CELL_SIZE,
                                             fill=color, outline="black")

        # Подсчет плотности машин на клетке
        density = {}
        for v in self.vehicles:
            key = (v.x, v.y)
            density[key] = density.get(key, 0) + 1

        # Рисуем транспортные средства с учетом плотности
        for v in self.vehicles:
            x0 = v.x * CELL_SIZE + 5
            y0 = v.y * CELL_SIZE + 5
            x1 = (v.x + 1) * CELL_SIZE - 5
            y1 = (v.y + 1) * CELL_SIZE - 5
            count = density[(v.x, v.y)]
            intensity = min(255, 50 + count * 40)
            color = v.color
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
            # Отображаем название машины и тип
            vehicle_label = f"{v.name} ({v.vehicle_type[0].upper()})"
            self.canvas.create_text(v.x * CELL_SIZE + CELL_SIZE // 2,
                                    v.y * CELL_SIZE + CELL_SIZE // 2,
                                    text=vehicle_label, fill="white", font=("Arial", 8))

    def update_info_display(self):
        self.info_label.config(text=f"Транспорт: {self.vehicle_count} | "
                                    f"Качество воздуха: {self.air_quality:.0f}% | "
                                    f"Скорость вентиляторов: {self.fans_speed:.0f}%")

    def calculate_air_quality(self):
        total_emissions = sum(v.emission_rate for v in self.vehicles)
        pollution = total_emissions * 0.8
        purification = self.fans_speed * 0.6
        base_degradation = 2.5
        if self.air_quality > 75:
            degradation_factor = 1.5
        elif self.air_quality < 50:
            degradation_factor = 0.5
        else:
            degradation_factor = 1.0
        natural_degradation = base_degradation * degradation_factor
        random_variation = random.uniform(-1.5, 1.5)
        delta = purification - pollution - natural_degradation + random_variation
        self.air_quality = max(0, min(100, self.air_quality + delta))
        return self.air_quality

    def apply_ventilation_rules(self, log_lines):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT condition, action, priority FROM ventilation_rules ORDER BY priority DESC")
        rules = c.fetchall()
        conn.close()

        air_quality = self.air_quality
        vehicle_count = self.vehicle_count
        fans_speed = self.fans_speed

        rules_applied = []
        for cond_text, action_text, priority in rules:
            try:
                if eval(cond_text):
                    old_speed = fans_speed
                    exec(action_text)
                    if fans_speed != old_speed:
                        rules_applied.append(f"Правило (приор.{priority})")
                        log_lines.append(f"  {cond_text[:40]}... → скорость {old_speed:.0f}→{fans_speed:.0f}%")
            except Exception as e:
                log_lines.append(f"  Ошибка: {e}")

        self.fans_speed = fans_speed
        if not rules_applied:
            log_lines.append(f"  Правила не сработали, скорость: {self.fans_speed:.0f}%")

    def fuzzy_ventilation_control(self, log_lines):
        try:
            self.ventilation_sim.input['vehicle_count'] = self.vehicle_count
            self.ventilation_sim.input['air_quality'] = self.air_quality
            self.ventilation_sim.compute()
            fuzzy_speed = self.ventilation_sim.output['fans_speed']
            if self.vehicle_count > 5 or self.air_quality < 50:
                if fuzzy_speed > self.fans_speed:
                    self.fans_speed = min(100, self.fans_speed + (fuzzy_speed - self.fans_speed) * 0.3)
            elif self.vehicle_count < 3 and self.air_quality > 70:
                pass
            else:
                self.fans_speed = self.fans_speed * 0.8 + fuzzy_speed * 0.2
            self.fans_speed = max(0, min(100, self.fans_speed))
            log_lines.append(f"  Нечеткое управление: fuzzy={fuzzy_speed:.1f}%, итого={self.fans_speed:.1f}%")
        except Exception as e:
            log_lines.append(f"  Ошибка нечеткого управления: {e}")

    def move_vehicles(self, log_lines):
        for vehicle in self.vehicles:
            if vehicle.status == "moving":
                if vehicle.x < TUNNEL_WIDTH - 1:
                    vehicle.x += 1
                else:
                    vehicle.status = "exiting"
        removed = len([v for v in self.vehicles if v.status == "exiting"])
        if removed > 0:
            log_lines.append(f"  {removed} транспортных средств покинуло туннель")
        self.vehicles = [v for v in self.vehicles if v.status == "moving"]
        self.vehicle_count = len(self.vehicles)

    def spawn_vehicles(self, log_lines):
        spawn_prob = 0.40
        if random.random() < spawn_prob:
            spawn_y = random.choice([1, 2, 3, 4])
            vehicle_type = random.choice(["car", "truck", "motorcycle"])
            if vehicle_type == "car":
                color = "blue"
            elif vehicle_type == "truck":
                color = "red"
            else:
                color = "green"
            new_vehicle = Vehicle(f"V{len(self.vehicles) + 1}", 0, spawn_y, color, vehicle_type)
            self.vehicles.append(new_vehicle)
            self.vehicle_count = len(self.vehicles)
            log_lines.append(f"  Новое ТС въехало: {vehicle_type} ({color})")

    def step(self):
        self.step_count += 1
        log_lines = [f"--- Шаг {self.step_count} ---"]

        old_fans_speed = self.fans_speed
        self.spawn_vehicles(log_lines)
        self.move_vehicles(log_lines)
        old_quality = self.air_quality
        self.calculate_air_quality()
        log_lines.append(
            f"До управления: ТС={self.vehicle_count}, возд={old_quality:.1f}%, вент={self.fans_speed:.1f}%")
        self.apply_ventilation_rules(log_lines)
        self.fuzzy_ventilation_control(log_lines)
        speed_change = self.fans_speed - old_fans_speed
        if abs(speed_change) > 0.1:
            change_sign = "+" if speed_change > 0 else ""
            log_lines.append(f"Изменение скорости: {change_sign}{speed_change:.1f}%")
        quality_change = self.air_quality - old_quality
        change_sign = "+" if quality_change > 0 else ""
        log_lines.append(
            f"ИТОГО: {self.vehicle_count} ТС, качество {self.air_quality:.1f}% ({change_sign}{quality_change:.1f}%), "
            f"вентиляция {self.fans_speed:.1f}%")

        self.draw_tunnel()
        self.update_info_display()
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "\n".join(log_lines))
        self.log_text.see(tk.END)

    def reset(self):
        self.vehicles = []
        self.create_initial_vehicles()
        self.air_quality = 45
        self.fans_speed = 35
        self.vehicle_count = len(self.vehicles)
        self.step_count = 0
        self.draw_tunnel()
        self.update_info_display()
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Система сброшена\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = TunnelVentilationApp(root)
    root.mainloop()