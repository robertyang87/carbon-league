"""User B: model removed with explicit permission. Original source preserved.
This derivative is tested separately; behavioral equivalence is not presumed.
"""
import numpy as np
from numpy.random.mtrand import random
from zerosum_env.envs.carbon.helpers import RecrtCenterAction, WorkerAction, Board
BaseActions = [None, RecrtCenterAction.RECCOLLECTOR, RecrtCenterAction.RECPLANTER]
WorkerActions = [None, WorkerAction.UP, WorkerAction.RIGHT, WorkerAction.DOWN, WorkerAction.LEFT]
WorkerDirections = np.stack([np.array((0, 0)), np.array((0, 1)), np.array((1, 0)), np.array((0, -1)), np.array((-1, 0))])

class ObservationParser:

    def __init__(self):
        self.reset()

    def reset(self):
        self.carbon_map = np.ones(shape=[15, 15], dtype=np.float32) * -1
        for x, y in [(10, 4), (4, 4), (4, 10), (10, 10)]:
            self.carbon_map[x, y] = 0
        self.last_pos_to_worker = np.zeros(shape=[15, 15], dtype='<U30')
        self.last_oppo_collector_carbon_map = np.zeros(shape=[15, 15], dtype=np.float32)
        self.cnt = 4

    def get_golbal_features(self, current_obs: Board, obs):
        map_size = [15, 15]
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        pos_to_worker = np.zeros(shape=[15, 15], dtype='<U30')
        local_carbon_map = np.zeros(map_size, dtype=np.float32)
        for point, cell in current_obs.cells.items():
            if cell.carbon > 0:
                local_carbon_map[point.x, point.y] = cell.carbon
        self.my_worker_map = np.zeros(map_size, dtype=np.int32)
        self.oppo_worker_map = np.zeros(map_size, dtype=np.int32)
        self.oppo_trees = current_obs.opponents[0].trees
        self.my_trees = current_obs.current_player.trees
        self.my_tree_map = np.zeros(map_size, dtype=np.int32)
        self.oppo_tree_map = np.zeros(map_size, dtype=np.int32)
        self.oppo_collector_carbon_map = np.zeros(map_size, dtype=np.float32)
        self.my_collector_carbon_map = np.zeros(map_size, dtype=np.float32)
        self.oppo_worker = current_obs.opponents[0].workers
        self.oppo_tree_eight_feature = np.zeros(map_size, dtype=np.float32)
        self.four_tree_nums_map = np.zeros([15, 15], dtype=np.int16)
        self.my_bases = [recrt.position for recrt in current_obs.current_player.recrtCenters]
        self.oppo_bases = [recrt.position for recrt in current_obs.opponents[0].recrtCenters]
        self.oppo_planter_count = 0
        self.my_tree_count = len(current_obs.current_player.trees)
        self.oppo_tree_count = len(current_obs.opponents[0].trees)
        self.updated_pos = set()
        tree_count = self.my_tree_count + self.oppo_tree_count
        self.buy_tree_cost = 20
        if self.my_tree_count > 0:
            self.buy_tree_cost = 20 + 5 * 1.235 ** tree_count
        for worker_id, worker in current_obs.workers.items():
            is_myself = worker.player_id == current_obs.current_player_id
            x, y = (worker.position.x, worker.position.y)
            pos_to_worker[x, y] = worker.id
            if is_myself:
                if worker.is_collector:
                    self.my_worker_map[x, y] = 1
                    self.carbon_map[x, y] = local_carbon_map[x, y]
                    self.updated_pos.add((x, y))
                    self.updata_other_pos(x, y, is_add=True)
                else:
                    self.my_worker_map[x, y] = 2
                    for i in range(4):
                        nx = (x + dx[i]) % 15
                        ny = (y + dy[i]) % 15
                        self.carbon_map[nx, ny] = local_carbon_map[nx, ny]
                        self.updated_pos.add((nx, ny))
                        self.updata_other_pos(nx, ny, is_add=True)
                self.my_collector_carbon_map[x, y] = worker.carbon
            else:
                self.oppo_worker_map[x, y] = 1 if worker.is_collector else 2
                self.oppo_collector_carbon_map[x, y] = worker.carbon
                if worker.is_collector and self.last_pos_to_worker[x, y] == pos_to_worker[x, y]:
                    self.carbon_map[x, y] *= 0.7
                    self.updated_pos.add((x, y))
                    inc = self.oppo_collector_carbon_map[x, y] - self.last_oppo_collector_carbon_map[x, y]
                    if inc > 0:
                        self.carbon_map[x, y] = inc / 0.3 * 0.7
                        self.updated_pos.add((x, y))
                if worker.is_planter:
                    self.oppo_planter_count += 1
                    if self.last_pos_to_worker[x, y] == pos_to_worker[x, y]:
                        self.updated_pos.add((x, y))
        for tree in current_obs.trees.values():
            x, y = (tree.position.x, tree.position.y)
            if tree.player_id == current_obs.current_player_id:
                self.my_tree_map[x, y] = 50 - tree.age
            else:
                self.oppo_tree_map[x, y] = 50 - tree.age
                self.oppo_tree_eight_feature[x, y] += 4 / 8
                for ddx in [x - 1, x, x + 1]:
                    for ddy in [y - 1, y, y + 1]:
                        self.oppo_tree_eight_feature[ddx % 15, ddy % 15] += 1
            if tree.age < 50:
                self.carbon_map[x, y] = 0
                self.updated_pos.add((x, y))
            else:
                self.carbon_map[x, y] = 30 / 1.03
            for i in range(4):
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                self.four_tree_nums_map[nx, ny] += 1
                if self.my_worker_map[x, y] != 1 and self.oppo_worker_map[x, y] != 1:
                    if tree.age > 1:
                        self.carbon_map[nx, ny] *= 1 - 0.0375
                        self.updated_pos.add((nx, ny))
        for x in range(15):
            for y in range(15):
                if self.carbon_map[x, y] > 0 and (x, y) not in self.updated_pos:
                    self.carbon_map[x, y] *= 1.03
                    self.carbon_map[x, y] = min(100, self.carbon_map[x, y])
        self.plant_value_map = np.zeros([15, 15], dtype=np.float32)
        for x in range(15):
            for y in range(15):
                if (x, y) in [(4, 4), (4, 10), (10, 4), (10, 10)]:
                    continue
                k = 0
                for i in range(4):
                    nx = (x + dx[i]) % 15
                    ny = (y + dy[i]) % 15
                    carbon = self.carbon_map[nx, ny] if self.carbon_map[nx, ny] >= 0 else 20
                    self.plant_value_map[x, y] += carbon / (1 + self.four_tree_nums_map[nx, ny])
                    if self.my_tree_map[nx, ny] > 0:
                        k += 1
                    if self.oppo_tree_map[nx, ny] > 0:
                        k -= 1
                if self.carbon_map[x, y] > 40:
                    if k < 0:
                        self.plant_value_map[x, y] *= 1.5
                    if k > 0:
                        self.plant_value_map[x, y] /= 1.5
        self.future_value_map = np.zeros(shape=[15, 15, 4], dtype=np.float32)
        for i in range(15):
            for j in range(15):
                self.future_value_map[i, j, :] = self.Future_value_calculation(i, j)
        self.theta = sorted(self.carbon_map.reshape(-1), key=lambda x: -x)[6]
        self.theta2 = max(sorted(self.carbon_map.reshape(-1), key=lambda x: -x)[50], 1)
        self.last_pos_to_worker = pos_to_worker.copy()
        self.last_oppo_collector_carbon_map = self.oppo_collector_carbon_map.copy()
        features = [self.my_worker_map, self.oppo_worker_map, self.my_tree_map, self.oppo_tree_map, self.oppo_collector_carbon_map, self.my_collector_carbon_map, self.oppo_tree_eight_feature, self.four_tree_nums_map, self.carbon_map, self.plant_value_map]
        features = np.array(features).reshape(-1).copy() / 1000
        features = np.hstack([features, self.my_tree_count / 20, self.oppo_tree_count / 20])
        return features

    def base_features(self, step, p_count, c_count):
        if step < 40:
            top_k = 30
        else:
            top_k = 15
        plant_value_list = sorted(self.plant_value_map.reshape(-1).copy(), key=lambda x: -x)
        p_value = plant_value_list[top_k]
        cnt = 0
        for x in range(15):
            for y in range(15):
                if self.my_worker_map[x, y] == 2 and self.plant_value_map[x, y] > p_value:
                    cnt += 1
        w = p_count + c_count
        distinctive_feature = 0
        if step > 50:
            if plant_value_list[5] < 120 and self.cnt < 6:
                self.cnt += 2
            elif plant_value_list[5] > 120 and self.cnt >= 6:
                self.cnt -= 2
            if plant_value_list[10] > 120 and self.cnt >= 4:
                self.cnt -= 2
            elif plant_value_list[10] < 120 and self.cnt < 4:
                self.cnt += 2
        L = [1, 2, 1, 1, 2, 2, 1, 2, 1, 1]
        if (self.my_cash >= self.buy_tree_cost * min(p_count, cnt + 1) + 30 and step < 283 or (c_count < 2 and self.my_cash >= 30)) and w < 10:
            if step < 40:
                distinctive_feature = L[w]
            elif p_count < c_count - self.cnt:
                distinctive_feature = 2
            else:
                distinctive_feature = 1
        if step == 0:
            if p_count == 0:
                distinctive_feature = 2
            else:
                distinctive_feature = 1
        feature_weight = [0, 0, 0, 0, 0]
        feature_weight[distinctive_feature] = 1
        this_feature = [distinctive_feature, 0]
        if this_feature[0] > 0:
            self.my_cash -= 30
        return (this_feature, feature_weight)

    def Future_value_calculation(self, px, py):
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        future_weight = [0, 0, 0, 0]
        deep = 3

        def dfs(x: int, y: int, t: int, back: int, h: int, c: float) -> None:
            if h == deep:
                future_weight[(t + 2) % 4] = max(future_weight[(t + 2) % 4], c)
                return
            for i in range(4):
                if i == t or i == back:
                    continue
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                dfs(nx, ny, t, (i + 2) % 4, h + 1, c + self.carbon_map[nx, ny] - self.oppo_tree_eight_feature[nx, ny] * 0.1 * c * 2 - int(self.oppo_tree_map[nx, ny] > h) * 2 * c)
        x, y = (px, py)
        for i in range(4):
            nx = (x + dx[i]) % 15
            ny = (y + dy[i]) % 15
            dfs(x=nx, y=ny, t=(i + 2) % 4, back=(i + 2) % 4, h=1, c=self.carbon_map[nx, ny])
        return future_weight

    def get_back_features(self, x, y, my_base):
        incx, incy = (-1, -1)
        if (abs(x - my_base.x) <= abs(15 - abs(x - my_base.x))) ^ (x <= my_base.x):
            incx = 1
        if (abs(y - my_base.y) <= abs(15 - abs(y - my_base.y))) ^ (y <= my_base.y):
            incy = 1
        if x == my_base.x:
            incx = 0
        if y == my_base.y:
            incy = 0
        nx = my_base.x
        fx = False
        dp = np.zeros([15, 15], dtype=np.float32)
        while True:
            if fx:
                break
            ny = my_base.y
            fy = False
            while True:
                if fy:
                    break
                dis = self.dis(x, y, nx, ny)
                dp[nx, ny] = max(dp[(nx - incx) % 15, ny], dp[nx, (ny - incy) % 15]) + max(0, self.carbon_map[nx, ny]) * 0.9 ** dis
                if self.oppo_tree_map[nx, ny] > dis:
                    dp[nx, ny] = 0
                fy = ny == y
                ny = (ny + incy) % 15
            fx = nx == x
            nx = (nx + incx) % 15
        dir = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
        feature_weight = [0, 0, 0, 0]
        if incx != 0:
            feature_weight[dir[-incx, 0]] = dp[(x - incx) % 15, y] + 0.01
        if incy != 0:
            feature_weight[dir[0, -incy]] = dp[x, (y - incy) % 15] + 0.01
        s = sum(feature_weight)
        for i in range(4):
            feature_weight[i] /= s + 1e-05
        return feature_weight

    def updata_other_pos(self, x, y, restore=False, is_add=False):
        for xx in [x, 14 - x]:
            for yy in [y, 14 - y]:
                if xx == x and yy == y:
                    continue
                if self.carbon_map[xx, yy] < 0:
                    self.carbon_map[xx, yy] = self.carbon_map[x, y]
                    if is_add:
                        self.updated_pos.add((xx, yy))
                    if restore:
                        self.restore_pos.add((xx, yy))

    def dis(self, x1, y1, x2, y2):
        return min(abs(x1 - x2), abs(15 - abs(x1 - x2))) + min(abs(y1 - y2), abs(15 - abs(y1 - y2)))

    def collector_feature(self, w, step, w_count, obs):
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        feature_weight = [0, 0, 0, 0, 0]

        def is_dangerous(x, y, my_c):
            for i in range(4):
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                if self.oppo_worker_map[nx, ny] == 1 and self.oppo_collector_carbon_map[nx, ny] < my_c:
                    return True
            if self.oppo_worker_map[x, y] == 1 and self.oppo_collector_carbon_map[x, y] < my_c:
                return True
            return False
        deep = 6
        c_map = self.carbon_map.copy()
        t_map = np.zeros([15, 15], dtype=np.int16)
        f_pow = np.array([1.03 ** t for t in range(10)])

        def dfs(x: int, y: int, t: int, is_stop: int, first_ac: int, last_ac: int, c: float) -> None:
            if first_ac != -1:
                future = 0
                if t >= 5:
                    ox, oy = (w.position.x, w.position.y)
                    ddx = 1
                    if (abs(x - ox) < abs(15 - abs(x - ox))) ^ (x <= ox):
                        future = max(future, self.future_value_map[x, y, ddx])
                    if (abs(x - ox) < abs(15 - abs(x - ox))) ^ (x >= ox):
                        future = max(future, self.future_value_map[x, y, ddx + 2])
                    ddy = 0
                    if (abs(y - oy) < abs(15 - abs(y - oy))) ^ (y <= oy):
                        future = max(future, self.future_value_map[x, y, ddy])
                    if (abs(y - oy) < abs(15 - abs(y - oy))) ^ (y >= oy):
                        future = max(future, self.future_value_map[x, y, ddy + 2])
                    future *= 0.1
                feature_weight[first_ac] = max(feature_weight[first_ac], c + future)
            if t >= deep:
                return
            c_last, t_last = (c_map[x, y], t_map[x, y])
            if c_last < 0:
                c_last = 30
            if self.my_worker_map[x, y] == 1:
                c_now = 0
            else:
                c_now = min(100, c_last * f_pow[t - t_last - is_stop])
            absorbed_c = c_now * 0.3
            c_now -= absorbed_c
            c_map[x, y], t_map[x, y] = (c_now, t)
            if not (first_ac == -1 and is_dangerous(x, y, c)) and (c_map[x, y] > 15 or c_map[x, y] > self.theta2):
                des_rate = 1 - 0.2 * self.oppo_tree_eight_feature[x, y]
                dfs(x=x, y=y, t=t + 1, is_stop=1, first_ac=0 if first_ac == -1 else first_ac, last_ac=0, c=c * des_rate + absorbed_c)
            c_map[x, y], t_map[x, y] = (c_last, t_last)
            for i in range(4):
                if (i + 2) % 4 + 1 == last_ac:
                    continue
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                if first_ac == -1 and is_dangerous(nx, ny, c):
                    continue
                if self.oppo_tree_map[nx, ny] > t:
                    continue
                va = 0
                if first_ac == -1 and self.oppo_worker_map[nx, ny] == 1 and (self.oppo_collector_carbon_map[nx, ny] > c):
                    va = c_map[nx, ny] * 0.1
                if self.oppo_worker_map[nx, ny] == 2 and first_ac == -1:
                    for j in range(4):
                        nnx = (nx + dx[j]) % 15
                        nny = (ny + dy[j]) % 15
                        va += c_map[nnx, nny] * 0.1 / 4
                des_rate = 1 - 0.2 * self.oppo_tree_eight_feature[nx, ny]
                dfs(x=nx, y=ny, t=t + 1, is_stop=0, first_ac=i + 1 if first_ac == -1 else first_ac, last_ac=i + 1, c=c * des_rate + va)
        x, y = (w.position.x, w.position.y)
        self.my_worker_map[x, y] = 0
        dfs(x=x, y=y, t=0, is_stop=0, first_ac=-1, last_ac=0, c=w.carbon)
        self.my_worker_map[x, y] = 1
        is_go_home = False
        home_weight = [0, 0]
        for i in range(2):
            hx, hy = (self.my_bases[i].x, self.my_bases[i].y)
            d = self.dis(x, y, hx, hy)
            if d == 0:
                home_weight[i] = 100
                continue
            co = 0
            for xx in [-2, -1, 0, 1, 2]:
                for yy in [-2, -1, 0, 1, 2]:
                    if abs(xx) + abs(yy) <= 2:
                        nx = (hx + xx) % 15
                        ny = (hy + yy) % 15
                        if self.oppo_worker_map[nx, ny] == 1 and self.carbon_map[nx, ny] < w.carbon + 30:
                            co += 1
            home_weight[i] = 1 / d / (co ** 2 + 2)
        home = self.my_bases[np.argmax(home_weight)]
        dd = self.dis(x, y, home.x, home.y)
        max_carbon = 0
        for i in range(4):
            nx = (x + dx[i]) % 15
            ny = (y + dy[i]) % 15
            max_carbon = max(max_carbon, self.carbon_map[nx, ny])
        home_throd = 450 if max_carbon < 70 else 550
        if w.carbon >= home_throd or (300 - step - 3 - dd / 4 <= dd and w.carbon > 10) or (w_count < 9 and w.carbon > 40 and (step <= 55)) or (is_dangerous(x, y, w.carbon) and w.carbon > 80) or (step > 275 and w.carbon > 150) or (self.my_cash < 150 and w.carbon > 75) or (dd <= 3 and w.carbon > 100):
            if 300 - step >= dd:
                get_back_weight = self.get_back_features(x, y, home)
                for i in range(4):
                    nx = (x + dx[i]) % 15
                    ny = (y + dy[i]) % 15
                    if is_dangerous(nx, ny, w.carbon) and (nx, ny) not in self.my_bases:
                        continue
                    is_go_home = True
                    feature_weight[i + 1] += get_back_weight[i] * 2000 * deep
        if is_go_home and self.my_cash > 30 or not is_go_home:
            if 300 - step - 3 - dd / 4 >= dd and (not is_dangerous(x, y, w.carbon)):
                if (w_count < 5 and self.carbon_map[x, y] >= 10 + is_go_home * 20 or (w_count < 8 and self.carbon_map[x, y] > 15 + is_go_home * 20)) and (step < 45 or step > 285):
                    feature_weight[0] += 2000 * deep
        s = sum(feature_weight)
        if s == 0:
            for i in range(4):
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                if not is_dangerous(nx, ny, w.carbon):
                    feature_weight[i + 1] = 1
            if not is_dangerous(x, y, w.carbon):
                feature_weight[0] = 0.5
            if sum(feature_weight) == 0:
                if self.my_tree_map[x, y] > 0:
                    feature_weight[0] += 10
                for i in range(4):
                    nx = (x + dx[i]) % 15
                    ny = (y + dy[i]) % 15
                    if self.oppo_worker_map[nx, ny] == 1:
                        feature_weight[i + 1] += 1 - int(self.carbon_map[nx, ny] > 30)
                    if self.my_tree_map[nx, ny] > 0:
                        feature_weight[i + 1] += 10
                    feature_weight[i + 1] += 1 / (2 + self.carbon_map[nx, ny])
        s = sum(feature_weight)
        for i in range(len(feature_weight)):
            feature_weight[i] /= s + 1e-05
        feature_weight_ = sorted([(v, idx) for idx, v in enumerate(feature_weight)], key=lambda x: -x[0])
        this_feature = [idx for v, idx in feature_weight_]
        return (this_feature, feature_weight)

    def collector_feature_v2(self, w, c_count):
        x, y = (w.position.x, w.position.y)
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        feature_weight = [0, 0, 0, 0]
        deep = 4
        local_c_dis = [9999]
        vis = set()

        def dfs(xx, yy, d):
            if d == deep or (xx, yy) in vis:
                return
            vis.add((xx, yy))
            if self.my_worker_map[xx, yy] == 1 and (xx != x and yy != y):
                local_c_dis[0] = min(local_c_dis[0], d)
            for i in range(4):
                nx = (xx + dx[i]) % 15
                ny = (yy + dy[i]) % 15
                dfs(nx, ny, d + 1)
        for ox in range(15):
            for oy in range(15):
                d = self.dis(x, y, ox, oy)
                if d == 0:
                    continue
                vis.clear()
                local_c_dis[0] = 9999
                dfs(ox, oy, 0)
                if local_c_dis[0] - 1 < d:
                    continue
                c = 0
                if self.oppo_worker_map[ox, oy] == 2:
                    c += 100
                    if self.oppo_tree_map[ox, oy] > 0:
                        c += 100
                c += self.carbon_map[ox, oy] * 10
                c /= d + 15
                if x != ox:
                    ddx = 1
                    if (abs(x - ox) < abs(15 - abs(x - ox))) ^ (x > ox):
                        feature_weight[ddx] = max(feature_weight[ddx], c)
                    else:
                        feature_weight[ddx + 2] = max(feature_weight[ddx + 2], c)
                if y != oy:
                    ddy = 0
                    if (abs(y - oy) < abs(15 - abs(y - oy))) ^ (y > oy):
                        feature_weight[ddy] = max(feature_weight[ddy], c)
                    else:
                        feature_weight[ddy + 2] = max(feature_weight[ddy + 2], c)
        s = sum(feature_weight)
        for i in range(len(feature_weight)):
            feature_weight[i] /= s + 1e-05
        feature_weight = [0] + feature_weight
        if self.carbon_map[x, y] > 65 or self.carbon_map[x, y] > self.theta:
            feature_weight[0] += 1
        this_feature = sorted([(v, idx) for idx, v in enumerate(feature_weight)], key=lambda x: -x[0])
        this_feature = [v for k, v in this_feature]
        return (this_feature, feature_weight)

    def planter_explore_feature(self, w, step, c_count):
        feature_weight = [0.01, 0.01, 0.01, 0.01]
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        lx, ux = (0, 8)
        ly, uy = (0, 8)
        deep = 5

        def dfs(x: int, y: int, t: int, back: int, h: int, se: set) -> None:
            if x < lx or x >= ux or y < ly or (y >= uy) or (h == deep):
                return
            if t != -1:
                feature_weight[t] = max(feature_weight[t], len(se))
            for i in range(4):
                if i == back:
                    continue
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                new_se = se.copy()
                for j in range(4):
                    nnx = (nx + dx[j]) % 15
                    nny = (ny + dy[j]) % 15
                    if self.carbon_map[nnx, nny] < 0:
                        new_se.add((nnx, nny))
                dfs(nx, ny, i if t == -1 else t, (i + 2) % 4, h + 1, new_se)
        x, y = (w.position.x, w.position.y)
        if x > 7:
            lx, ux = (7, 15)
        if y > 7:
            ly, uy = (7, 15)
        dfs(x=x, y=y, t=-1, back=-1, h=1, se=set())
        feature_weight = [0] + feature_weight
        va = 0
        for i in range(4):
            nx = (x + dx[i]) % 15
            ny = (y + dy[i]) % 15
            assert self.carbon_map[nx, ny] >= 0
            va += self.carbon_map[nx, ny]
        if self.my_cash < 30 and self.my_tree_count > 0:
            d1 = 100
            tree = self.my_trees[0].position
            tx, ty = (tree.x, tree.y)
            for w in self.oppo_worker:
                if w.is_planter:
                    w = w.position
                    d1 = min(d1, self.dis(tx, ty, w.x, w.y))
            d2 = self.dis(x, y, tx, ty)
            flag = 0
            for i in range(15):
                for j in range(15):
                    if self.my_worker_map[i, j] == 2:
                        if i == x and j == y:
                            continue
                        if self.dis(i, j, tx, ty) < d2:
                            flag = 1
            if flag == 0 and self.plant_value_map[tx, ty] > 50 / 2 and (c_count == 0):
                if d2 > 0 and d1 - d2 <= 1:
                    get_back_feature = [0] + self.get_back_features(x, y, tree)
                    for i in range(len(feature_weight)):
                        feature_weight[i] = feature_weight[i] + get_back_feature[i] * 12 * deep
                elif d2 == 0 and d1 <= 2:
                    feature_weight[0] += 200
        elif va >= 80 and self.my_cash >= self.buy_tree_cost and ((x, y) not in [(4, 4), (4, 10), (10, 4), (10, 10)]) and (self.my_tree_map[x, y] == 0):
            feature_weight[0] += 100

        def is_dangerous(xx, yy):
            if self.oppo_worker_map[xx, yy] == 1:
                return True
            for i in range(4):
                nx = (xx + dx[i]) % 15
                ny = (yy + dy[i]) % 15
                if self.oppo_worker_map[nx, ny] == 1:
                    return True
            return False
        s = sum(feature_weight)
        for i in range(len(feature_weight)):
            feature_weight[i] /= s + 1e-05
            if i == 0 and is_dangerous(x, y):
                feature_weight[i] /= 20
            if i > 0:
                nx = (x + dx[i - 1]) % 15
                ny = (y + dy[i - 1]) % 15
                if is_dangerous(nx, ny):
                    feature_weight[i] /= 20
        feature_weight_ = sorted([(v, idx) for idx, v in enumerate(feature_weight)], key=lambda x: -x[0])
        this_feature = [idx for v, idx in feature_weight_]
        if this_feature[0] != 0:
            nx = (x + dx[this_feature[0] - 1]) % 15
            ny = (y + dy[this_feature[0] - 1]) % 15
            for i in range(4):
                nnx = (nx + dx[i]) % 15
                nny = (ny + dy[i]) % 15
                if self.carbon_map[nnx, nny] < 0:
                    self.carbon_map[nnx, nny] = 10
                    self.restore_pos.add((nnx, nny))
                    self.updata_other_pos(nnx, nny, restore=True)
        return (this_feature, feature_weight)

    def planter_feature(self, w, step, c_count):
        x, y = (w.position.x, w.position.y)
        p_value = sorted(self.plant_value_map.reshape(-1).copy(), key=lambda x: -x)
        all_tree_count = self.my_tree_count + self.oppo_tree_count
        if all_tree_count < 6:
            top_k = 20
        elif all_tree_count < 10:
            top_k = 15
        else:
            top_k = 10
        threshold = p_value[top_k]
        feature_weight = [0, 0, 0, 0]
        action_planter_count = [0, 0, 0, 0]
        for i in range(15):
            for j in range(15):
                if self.my_tree_map[i, j] == 0 and self.oppo_tree_map[i, j] == 0 and (self.plant_value_map[i, j] > threshold or self.plant_value_map[i, j] >= 160):
                    if i == 4 and j == 10 or (i == 10 and j == 4):
                        continue
                    v = self.plant_value_map[i, j] / (2 + self.dis(x, y, i, j))
                    if i != x:
                        ddx = 1
                        if (abs(x - i) < abs(15 - abs(x - i))) ^ (x > i):
                            feature_weight[ddx] = max(feature_weight[ddx], v)
                        else:
                            feature_weight[ddx + 2] = max(feature_weight[ddx + 2], v)
                    if j != y:
                        ddy = 0
                        if (abs(y - j) < abs(15 - abs(y - j))) ^ (y > j):
                            feature_weight[ddy] = max(feature_weight[ddy], v)
                        else:
                            feature_weight[ddy + 2] = max(feature_weight[ddy + 2], v)
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        vis = set()
        deep = 3

        def dfs(xx: int, yy: int, t: int, back: int, h: int) -> None:
            if (xx, yy) not in vis:
                vis.add((xx, yy))
                if self.my_worker_map[xx, yy] > 0:
                    action_planter_count[(t + 2) % 4] += (deep - h) * 2 + 1
            if h == deep:
                return
            for i in range(4):
                if i == t or i == back:
                    continue
                nx = (xx + dx[i]) % 15
                ny = (yy + dy[i]) % 15
                dfs(nx, ny, t, (i + 2) % 4, h + 1)
        for i in range(4):
            nx = (x + dx[i]) % 15
            ny = (y + dy[i]) % 15
            vis.clear()
            dfs(xx=nx, yy=ny, t=(i + 2) % 4, back=(i + 2) % 4, h=1)
        start_rob_tree = 1
        this_value = 0
        action_weight_rob = [0, 0, 0, 0]
        s = sum(feature_weight)
        if len(self.oppo_trees) >= start_rob_tree:
            action_weight_rob, this_value = self.rob_feature(w, step)
        rob_rate = 0.6 + this_value / 420
        for i in range(4):
            feature_weight[i] /= 1e-05 + s
            if len(self.oppo_trees) >= start_rob_tree:
                feature_weight[i] = (1 - rob_rate) * feature_weight[i] + rob_rate * action_weight_rob[i]
        is_dangerous = False
        for i in range(4):
            feature_weight[i] /= action_planter_count[i] + 1
            nx = (x + dx[i]) % 15
            ny = (y + dy[i]) % 15
            for j in range(4):
                nnx = (nx + dx[j]) % 15
                nny = (ny + dy[j]) % 15
                if self.oppo_worker_map[nnx, nny] == 1:
                    feature_weight[i] /= 50
                    feature_weight[i] -= 1e-06
            if self.oppo_worker_map[nx, ny] == 1:
                is_dangerous = True
                feature_weight[i] /= self.carbon_map[nx, ny] + 2
                feature_weight[i] -= 1e-06
        feature_weight = [0] + feature_weight
        if (self.plant_value_map[x, y] >= threshold or self.plant_value_map[x, y] > 180) and self.plant_value_map[x, y] * (1 - (1 - 0.0375) ** min(16, 300 - step)) > self.buy_tree_cost or (len(self.oppo_trees) >= start_rob_tree and this_value > 0):
            if not (is_dangerous or (x, y) in [(4, 4), (4, 10), (10, 4), (10, 10)]):
                feature_weight[0] += 1
            else:
                feature_weight[i] -= 1e-06
        elif self.my_tree_map[x, y] > 0:
            oppo_d = 9999
            for w in self.oppo_worker:
                if w.is_planter:
                    oppo_d = min(oppo_d, self.dis(x, y, w.position.x, w.position.y))
            if oppo_d < 6:
                value = 0
                for i in range(4):
                    nx = (x + dx[i]) % 15
                    ny = (y + dy[i]) % 15
                    value += self.carbon_map[nx, ny] * 0.0375
                co = 0
                for xx in [-3, -2, -1, 0, 1, 2, 3]:
                    for yy in [-3, -2, -1, 0, 1, 2, 3]:
                        if abs(xx) + abs(yy) <= 3:
                            nx = (x + xx) % 15
                            ny = (y + yy) % 15
                            if self.oppo_worker_map[nx, ny] == 2:
                                co += 1
                if value > 3.5 and (not (is_dangerous or (x, y) in [(4, 4), (4, 10), (10, 4), (10, 10)])) or (co > 0 and value > 1.875):
                    feature_weight[0] += 1
        if is_dangerous:
            home_weight = [0, 0]
            for i in range(2):
                hx, hy = (self.my_bases[i].x, self.my_bases[i].y)
                d = self.dis(x, y, hx, hy)
                if d == 0:
                    home_weight[i] = 100
                    continue
                co = 0
                for xx in [-2, -1, 0, 1, 2]:
                    for yy in [-2, -1, 0, 1, 2]:
                        if abs(xx) + abs(yy) <= 2:
                            nx = (hx + xx) % 15
                            ny = (hy + yy) % 15
                            if self.oppo_worker_map[nx, ny] == 1 and self.carbon_map[nx, ny] < w.carbon + 30:
                                co += 1
                home_weight[i] = 1 / d / (co ** 2 + 2)
            home = self.my_bases[np.argmax(home_weight)]
            get_back_weight = self.get_back_features(x, y, home)
            for i in range(4):
                nx = (x + dx[i]) % 15
                ny = (y + dy[i]) % 15
                for j in range(4):
                    nnx = (nx + dx[j]) % 15
                    nny = (ny + dy[j]) % 15
                    if self.oppo_worker_map[nnx, nny] == 1:
                        get_back_weight[i] = 0
                if (nx, ny) not in self.my_bases:
                    continue
                feature_weight[i + 1] += get_back_weight[i] * 10
        s = sum(feature_weight)
        for i in range(len(feature_weight)):
            feature_weight[i] /= s + 1e-05
        this_feature = sorted([(v, idx) for idx, v in enumerate(feature_weight)], key=lambda x: -x[0])
        this_feature = [v[1] for v in this_feature]
        return (this_feature, feature_weight)

    def rob_feature(self, w, step):
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        x, y = (w.position.x, w.position.y)
        feature_weight = [0, 0, 0, 0]
        this_value = 0
        sum_value = 0
        deep = 3
        local_p_dis = [100]
        vis = set()

        def dfs(xx, yy, d):
            if d == deep or (xx, yy) in vis:
                return
            vis.add((xx, yy))
            if self.my_worker_map[xx, yy] == 2 and (xx != x and yy != y):
                local_p_dis[0] = min(local_p_dis[0], d)
            for i in range(4):
                nx = (xx + dx[i]) % 15
                ny = (yy + dy[i]) % 15
                dfs(nx, ny, d + 1)
        for tree in self.oppo_trees:
            tx, ty = (tree.position.x, tree.position.y)
            d = self.dis(x, y, tx, ty)
            vis.clear()
            local_p_dis[0] = 100
            dfs(tx, ty, 0)
            if local_p_dis[0] < d:
                continue
            res_age = 50 - tree.age
            if res_age < d + 5 or res_age < 15:
                continue
            value = 0
            a = 0.0375
            for i in range(4):
                nx = (tx + dx[i]) % 15
                ny = (ty + dy[i]) % 15
                ca = self.carbon_map[nx, ny]
                if ca < 0:
                    ca = 25
                res = ca * (1 - a * self.four_tree_nums_map[nx, ny]) ** d
                value += res * (1 - (1 - a) ** min(res_age - d, 300 - step))
            if value < 20:
                continue
            if self.oppo_worker_map[tx, ty] == 2:
                value /= 2
            d_to_my_base = min(self.dis(tx, ty, self.my_bases[0].x, self.my_bases[0].y), self.dis(tx, ty, self.my_bases[1].x, self.my_bases[1].y))
            if d_to_my_base <= 2:
                value += 35 / d_to_my_base
            sum_value += value
            if x == tx and y == ty:
                this_value = value
            if tx != x:
                ddx = 1
                if (abs(x - tx) < abs(15 - abs(x - tx))) ^ (x > tx):
                    feature_weight[ddx] = max(value, feature_weight[ddx])
                else:
                    feature_weight[ddx + 2] = max(value, feature_weight[ddx + 2])
            if ty != y:
                ddy = 0
                if (abs(y - ty) < abs(15 - abs(y - ty))) ^ (y > ty):
                    feature_weight[ddy] = max(value, feature_weight[ddy])
                else:
                    feature_weight[ddy + 2] = max(value, feature_weight[ddy + 2])
        s = sum(feature_weight)
        for i in range(4):
            feature_weight[i] /= s + 0.001
        return (feature_weight, this_value)

    def move_worker(self, x, y, distinctive_feature) -> None:
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        tox = (x + dx[distinctive_feature]) % 15
        toy = (y + dy[distinctive_feature]) % 15
        self.my_worker_map[tox, toy] = self.my_worker_map[x, y]
        self.my_worker_map[x, y] = 0
        self.my_collector_carbon_map[tox, toy] = self.my_collector_carbon_map[x, y]
        self.my_collector_carbon_map[x, y] = 0

    def obs_transform(self, obs, configuration):
        current_obs = Board(obs, configuration)
        agent_obs = {}
        l_feature = {}
        step = current_obs.step
        if current_obs.step == 0:
            self.reset()
        self.restore_pos = set()
        g_features = self.get_golbal_features(current_obs, obs)
        c_count = len(current_obs.current_player.collectors)
        p_count = len(current_obs.current_player.planters)
        cash = current_obs.current_player.cash
        g_features = np.hstack([g_features, c_count / 10, p_count / 10, cash / 1000, step / 300])
        self.my_cash = cash
        my_workers = {w.id: w for w in current_obs.current_player.workers}
        agent_ids = [w.id for w in current_obs.current_player.workers] + [id.id for id in current_obs.current_player.recrtCenters]
        base_pos = {b.id: b.position for b in current_obs.current_player.recrtCenters}
        agent_ids_weight = []
        for k in agent_ids:
            dis = 0
            if k in my_workers:
                w = my_workers[k]
                dis = self.dis(w.position.x, w.position.y, self.oppo_bases[0].x, self.oppo_bases[0].y)
                dis = min(dis, self.dis(w.position.x, w.position.y, self.oppo_bases[1].x, self.oppo_bases[1].y))
                xx, yy = (w.position.x, w.position.y)
                if w.is_collector:
                    agent_ids_weight.append((k, 1000 + self.carbon_map[xx, yy]))
                else:
                    agent_ids_weight.append((k, self.plant_value_map[xx, yy] / 10))
            else:
                weight = 0
                x, y = (base_pos[k].x, base_pos[k].y)
                co = 1
                for xx in [-3, -2, -1, 0, 1, 2, 3]:
                    for yy in [-3, -2, -1, 0, 1, 2, 3]:
                        nx = (x + xx) % 15
                        ny = (y + yy) % 15
                        if self.carbon_map[nx, ny] > 0:
                            weight += self.carbon_map[nx, ny] / 10
                        if self.my_worker_map[nx, ny] > 0:
                            co += 1
                agent_ids_weight.append((k, weight / (co + 4)))
            if k not in l_feature:
                l_feature[k] = []
            l_feature[k].append(dis / 15)
        agent_ids_weight = sorted(agent_ids_weight, key=lambda x: -x[1])
        position = set()
        position.add((self.oppo_bases[0].x, self.oppo_bases[0].y))
        position.add((self.oppo_bases[1].x, self.oppo_bases[1].y))
        do_v = 0
        do_v2 = 0
        unkown_count = 0
        for i in range(15):
            for j in range(15):
                if self.carbon_map[i, j] < 0:
                    unkown_count += 1
        g_features = np.hstack([g_features, unkown_count / 15 / 15])
        min_res_age = 5
        for tree in current_obs.current_player.trees:
            min_res_age = min(50 - tree.age, min_res_age)
        for aid, weight in agent_ids_weight:
            if aid not in my_workers:
                this_feature, feature_weight = self.base_features(step, p_count, c_count)
            elif my_workers[aid].is_collector:
                if do_v >= (3 if step < 45 else 2) and do_v2 <= 4 and (my_workers[aid].carbon == 0):
                    this_feature, feature_weight = self.collector_feature_v2(my_workers[aid], c_count)
                    do_v2 += 1
                else:
                    this_feature, feature_weight = self.collector_feature(my_workers[aid], step, c_count + p_count, obs)
                    do_v += 1
            elif unkown_count > 40 and step < 22:
                this_feature, feature_weight = self.planter_explore_feature(my_workers[aid], step, c_count)
            else:
                this_feature, feature_weight = self.planter_feature(my_workers[aid], step, c_count)
            if aid in my_workers:
                x, y = (my_workers[aid].position.x, my_workers[aid].position.y)
            else:
                if aid == current_obs.current_player.recrtCenters[0].id:
                    posi = current_obs.current_player.recrtCenters[0].position
                else:
                    posi = current_obs.current_player.recrtCenters[1].position
                x, y = (posi.x, posi.y)
            dx = [0, 1, 0, -1]
            dy = [1, 0, -1, 0]
            distinctive_feature = 0
            nx, ny = (x, y)
            for ac in this_feature:
                if ac != 0 and aid in my_workers:
                    nx = (x + dx[ac - 1]) % 15
                    ny = (y + dy[ac - 1]) % 15
                if aid not in my_workers and ac == 0:
                    break
                if self.oppo_tree_map[nx, ny] > 0 and my_workers[aid].is_collector and (my_workers[aid].carbon > 20):
                    continue
                if (nx, ny) not in position:
                    if step < 292:
                        position.add((nx, ny))
                    distinctive_feature = ac
                    break
            feature_weight += np.eye(len(feature_weight))[distinctive_feature] * (1 + 0.0375) ** min_res_age
            l_feature[aid] += [x / 15, y / 15, do_v / 10, do_v2 / 10, *feature_weight, distinctive_feature]
            gg_features = np.hstack([g_features, l_feature[aid]])
            if distinctive_feature != 0:
                if 'recrtCenter' not in aid:
                    self.move_worker(my_workers[aid].position.x, my_workers[aid].position.y, distinctive_feature - 1)
                elif distinctive_feature == 1:
                    c_count += 1
                elif distinctive_feature == 2:
                    p_count += 1
            if aid not in my_workers:
                typ = 'base'
            elif my_workers[aid].is_collector:
                typ = 'COLLECTOR'
            else:
                typ = 'PLANTER'
            agent_obs[aid, typ] = gg_features.copy()
        for xx, yy in self.restore_pos:
            self.carbon_map[xx, yy] = -1
        return agent_obs

class MyPolicy:
    def __init__(self):
        self.obser = ObservationParser()
    def take_action(self, observation, configuration):
        features = self.obser.obs_transform(observation, configuration)
        commands = {}
        for (aid, typ), v in features.items():
            action = int(v[-1])
            if action:
                commands[aid] = (BaseActions if typ == 'base' else WorkerActions)[action].name
        return commands

my_policy = MyPolicy()
def agent(obs, configuration):
    return my_policy.take_action(obs, configuration)
