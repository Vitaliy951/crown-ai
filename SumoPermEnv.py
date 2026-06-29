import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Проверяем, настроен ли путь к SUMO в системе
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Пожалуйста, объявите переменную окружения 'SUMO_HOME'")

import traci  # Импортируем TraCI после проверки путей

class SumoPermEnv(gym.Env):
    def __init__(self, config_file="perm_city.sumocfg", use_gui=False):
        super(SumoPermEnv, self).__init__()
        
        self.config_file = config_file
        # use_gui=True откроет визуальное окно SUMO. False — быстрый расчет в консоли (для ЦОД)
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        
        # ID светофора в вашей схеме SUMO
        self.tls_id = "node_permskaya_intersection" 
        
        # Список ID входящих полос (дорог) к перекрестку
        self.controlled_lanes = ["lane_north_0", "lane_south_0", "lane_east_0", "lane_west_0"]
        self.num_lanes = len(self.controlled_lanes)

        # Пространства действий и наблюдений (как в прошлом шаге)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "queues": spaces.Box(low=0, high=100, shape=(self.num_lanes, 3), dtype=np.int32),
            "current_phase": spaces.Discrete(4)
        })

        # Приоритеты транспорта
        self.weight_cars = 1.0
        self.weight_trucks = 1.5
        self.weight_buses = 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Закрываем предыдущую сессию, если она была активна
        try: traci.close()
        except: pass

        # Запускаем SUMO через TraCI
        traci.start([self.sumo_binary, "-c", self.config_file, "--tripinfo-output", "tripinfo.xml"])
        
        self.current_phase = 0
        traci.trafficlight.setPhase(self.tls_id, self.current_phase)

        # Считываем стартовое состояние
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """ Ключевой метод: собирает реальные данные из SUMO """
        lane_queues = np.zeros((self.num_lanes, 3), dtype=np.int32)

        for idx, lane_id in enumerate(self.controlled_lanes):
            # Получаем список ID всех машин, стоящих в очереди на этой полосе
            # В SUMO машина считается стоящей, если её скорость ниже 0.1 м/с
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            
            for veh_id in vehicle_ids:
                # Проверяем, стоит ли машина (длина очереди)
                if traci.vehicle.getSpeed(veh_id) < 0.1:
                    # Фиксируем тип транспорта, заданный в XML-файле маршрутов
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    
                    if "bus" in veh_type:
                        lane_queues[idx][2] += 1  # Автобус
                    elif "truck" in veh_type:
                        lane_queues[idx][1] += 1  # Грузовик
                    else:
                        lane_queues[idx][0] += 1  # Легковая

        return {
            "queues": lane_queues,
            "current_phase": self.current_phase
        }

    def step(self, action):
        phase_changed = (action != self.current_phase)
        self.current_phase = action

        # 1. Отдаем команду светофору в SUMO
        traci.trafficlight.setPhase(self.tls_id, int(self.current_phase))

        # 2. Продвигаем симуляцию SUMO вперед на 5 секунд (5 шагов по 1 сек)
        # Это дает ИИ время оценить результат своего действия
        for _ in range(5):
            traci.simulationStep()

        # 3. Собираем новые данные из симулятора
        obs = self._get_observation()

        # 4. Считаем награду на основе реальных данных из SUMO
        total_weighted_queue = 0
        for lane_data in obs["queues"]:
            lane_loss = (lane_data[0] * self.weight_cars) + \
                        (lane_data[1] * self.weight_trucks) + \
                        (lane_data[2] * self.weight_buses)
            total_weighted_queue += lane_loss

        reward = -total_weighted_queue
        if phase_changed:
            reward -= 10.0  # Штраф за смену фазы

        # Проверяем, не закончились ли машины в симуляции вообще
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = False

        return obs, reward, terminated, truncated, {}

    def close(self):
        traci.close()
