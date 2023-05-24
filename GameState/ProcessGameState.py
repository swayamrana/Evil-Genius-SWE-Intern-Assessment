import pyarrow
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import math

class ProcessGameState:
    def __init__(self, game_data):
        self.state = pd.read_parquet(game_data, engine='pyarrow')
    def in_boundaries(self, x_bound, y_bound, z_bound):
        # Dictionary that iterates through each row and checks the location of the player fits the boundaries.
        boundary_dict = {}
        for i, row in self.state.iterrows():
            if(row['x'] in range(x_bound[0], x_bound[1]+1)
            and row['y'] in range(y_bound[0], y_bound[1]+1)
            and row['z'] in range(z_bound[0], z_bound[1]+1)):
                boundary_dict[i] = True
            else:
                boundary_dict[i] = False
        return boundary_dict

    def extract_weapons(self):
        # Dictionary with the key as the weapon class and the values as the count of each weapon in the inventories.
        weapon_dict = {}
        for inv_arr in self.state['inventory']:
            if inv_arr is not None:
                for weapon in inv_arr:
                    class_ = weapon['weapon_class']
                    cur_cnt = weapon_dict[class_] if class_ in weapon_dict else 0
                    weapon_dict[class_] = cur_cnt+1
        return weapon_dict

gs = ProcessGameState("game_state_frame_data.parquet")

# Q1: Is entering via the light blue boundary a common strategy used by Team2 on T (terrorist) side?
lightblue = gs.in_boundaries([-1735, 250], [-2806, 1233], [285, 421])
t2 = gs.state[(gs.state['team'] == 'Team2') & (gs.state['side'] == 'T')]
cnt = 0
sum = 0
for i, row in t2.iterrows():
    if lightblue[i]:
        cnt = cnt + 1
    sum = sum + 1
print("We find that when Team 2 is on T side, they only enter the light blue boundary " + str((round((cnt/sum) * 100, 3))) + "% of the time out of all rounds, not making it a common strategy.")

# Q2: What is the average timer that Team2 on T (terrorist) side enters “BombsiteB” with least 2 rifles or SMGs?
bb_weapons = 0
for _, row in t2.iterrows():
   if row["area_name"] == "BombsiteB":
        if row['inventory'] is not None:
            for inv_arr in row['inventory']:
                total_w = 0
                if inv_arr is not None:
                    for weapon_class in inv_arr:
                        if(weapon_class == "Rifle") or (weapon_class == "SMG"):
                            total_w = total_w + 1
                    if total_w >= 2:
                        bb_weapons = bb_weapons + 1
print("We see that there are " + str(bb_weapons) + " times that Team 2 on the T side enters BombsiteB with at least 2 rifles or SMGs, so we can deduce that the average timer will be 0 as well.")

#Q3: Now that we’ve gathered data on Team2 T side, let's examine their CT (counter-terrorist) Side. Using the same data set, tell our coaching staff where you suspect them to be waiting inside “BombsiteB”
# Plot a seaborn heatmap of the data that fits the criteria
team2_ct_side = gs.state[(gs.state['team'] == 'Team2') & (gs.state['side'] == 'CT')]
bb_heatmap_info = team2_ct_side[team2_ct_side['area_name'] == 'BombsiteB']
heatmap = seaborn.heatmap(bb_heatmap_info[["x", "y"]])

# Use a kernel density estimation to find the biggest clusters in our set of points(250 as a sample).
k = gaussian_kde([bb_heatmap_info['x'], bb_heatmap_info['y']])
x = np.linspace(min(bb_heatmap_info['x']), max(bb_heatmap_info['x']), 250)
y = np.linspace(min(bb_heatmap_info['y']), max(bb_heatmap_info['y']), 250)

# Apply this estimation onto our sample x and y points that are in range of our original set.
m_x, m_y = np.meshgrid(x, y)
gd = k(np.vstack([m_x.flatten(), m_y.flatten()]))

# Find the maximum index for each dimension and return the point with the maximum density.
i = np.argmax(gd)
m_x = m_x.flatten()
m_y = m_y.flatten()
print("We can expect them to be waiting inside Bombsite B at the point (" + str(math.ceil(m_x[i])) + "," + str(math.ceil(m_y[i])) + ")")
plt.show()
