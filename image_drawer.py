import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import random


class ImageDrawer:
    def __init__(self, champion_circle_icon_path, minimap_path,
                 fog_path, misc_path, resize=None):
        self.champion_circle_icon_path = champion_circle_icon_path
        self.minimap_path = minimap_path
        self.fog_path = fog_path
        self.misc_path = misc_path
        self.resize = resize

        self.setup()

    #################################################
    # Setup code
    #################################################

    def load_image(self, p):
        """Loads an image from a path

        Args:
            p (string): path of image

        Returns:
            (torch.tensor, string): (image in torch.tensor format, base name of image)
        """
        if os.path.isfile(p):
            full_p = os.path.basename(p)
            split = os.path.splitext(full_p)
            basefile = split[0]
            extension = split[1]
            if extension == '.png':
                img = Image.open(p).convert("RGBA")
                img = transforms.ToTensor()(img)
                return (img, basefile)
        return None

    def setup(self):
        self.champion_icons = []
        self.minimaps = []
        self.fogs = []
        self.miscs = []
        self.ally_circle = None
        self.enemy_circle = None

        # loads images from path into array
        def load_results(path, array):
            for p in os.listdir(path):
                p = os.path.join(path, p)
                result = self.load_image(p)
                if result:
                    img, label = result
                    array.append((img, label))

        load_results(self.champion_circle_icon_path, self.champion_icons)
        load_results(self.minimap_path, self.minimaps)
        load_results(self.fog_path, self.fogs)
        load_results(self.misc_path, self.miscs)
        self.champion_icons.sort(key=lambda x: x[1])

        self.minimap_size = tuple(self.minimaps[0][0].shape[1:3])
        self.minimap_ratio = (self.resize[1] / self.minimap_size[0], self.resize[0] / self.minimap_size[1])
        self.minimap_size = (int(self.minimap_size[0] * self.minimap_ratio[0]), int(self.minimap_size[1] * self.minimap_ratio[1]))

        self.id_to_champion = {}
        self.champion_to_id = {}
        int_champion = 1
        for c, label in self.champion_icons:
            b = c.numpy()
            b = b.transpose((2, 1, 0))
            h, w = b.shape[:2]
            b[b[..., 3] == 0] = (0.0, 0.0, 0.0, 0.0)
            b = b.transpose((2, 1, 0))
            c = torch.from_numpy(b)

            self.id_to_champion[int_champion] = label
            self.champion_to_id[label] = int_champion

            int_champion += 1

        for c, _ in self.miscs:
            b = c.numpy()
            b = b.transpose((2, 1, 0))
            h, w = b.shape[:2]
            b[b[..., 3] == 0] = (0.0, 0.0, 0.0, 0.0)
            b.transpose((2, 1, 0))
            c = torch.from_numpy(b)

        self.red_outer_towers = []
        self.red_normal_tower = None
        self.red_inhib = None
        self.red_nexus = None

        self.blue_outer_towers = []
        self.blue_normal_tower = None
        self.blue_inhib = None
        self.blue_nexus = None

        for m, label in self.miscs:
            if 'tower' in label or 'inhib' in label or 'nexus' in label:
                m = m.numpy()
                m = m.transpose((2, 1, 0))
                #m[m[..., 0] > 0.2] = (28/255, 79/255, 93/255, 1.0)
                lighter = m[..., 0] > 0.5
                darker = m[..., 0] > 0.2

                blue = m.copy()
                blue[darker] = (28/255, 79/255, 93/255, 1.0)
                blue[lighter] = (60/255, 167/255, 199/255, 1.0)

                red = m.copy()
                red[darker] = (92/255, 24/255, 25/255, 1.0)
                red[lighter] = (189/255, 52/255, 51/255, 1.0)

                red = red.transpose((2, 1, 0))
                blue = blue.transpose((2, 1, 0))

                m = m.transpose((2, 1, 0))

                if 'tower_minimap' in label:
                    red = torch.from_numpy(red)
                    blue = torch.from_numpy(blue)
                    self.red_outer_towers.append((red, label + '_red'))
                    self.blue_outer_towers.append((blue, label + '_blue'))
                    red = red.numpy()
                    blue = blue.numpy()
                if 'icon_ui_tower_minimap' == label:  # regular tower
                    red = torch.from_numpy(red)
                    blue = torch.from_numpy(blue)
                    self.red_normal_tower = (red, label + '_red')
                    self.blue_normal_tower = (blue, label + '_blue')
                elif 'inhib' in label:
                    red = torch.from_numpy(red)
                    blue = torch.from_numpy(blue)
                    self.red_inhib = (red, label + '_red')
                    self.blue_inhib = (blue, label + '_blue')
                elif 'nexus' in label:
                    red = red.transpose((2, 1, 0))
                    blue = blue.transpose((2, 1, 0))
                    scale = 1.5
                    red = cv2.resize(red, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    blue = cv2.resize(blue, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    red = red.transpose((2, 1, 0))
                    blue = blue.transpose((2, 1, 0))
                    red = torch.from_numpy(red)
                    blue = torch.from_numpy(blue)
                    self.red_nexus = (red, label + '_red')
                    self.blue_nexus = (blue, label + '_blue')

        def MakePos(x, y):
            m_x = 14640.673
            m_y = 14777.2646
            offset_x = 200  # -2000
            offset_y = -800
            result_x = (x - offset_x) / m_x
            result_y = (m_y - (y - offset_y)) / m_y
            return result_x, result_y

        # position of towers
        self.blue_tower_pos_bot_a = MakePos(10097.62, 808.73)
        self.blue_tower_pos_bot_b = MakePos(6512.53, 1262.62)
        self.blue_tower_pos_bot_c = MakePos(3747.26, 1041.04)

        self.blue_tower_pos_mid_a = MakePos(5448.02, 6169.10)
        self.blue_tower_pos_mid_b = MakePos(4657.66, 4591.91)
        self.blue_tower_pos_mid_c = MakePos(3233.99, 3447.24)

        self.blue_tower_pos_top_a = MakePos(574.66, 10220.47)
        self.blue_tower_pos_top_b = MakePos(1106.26, 6485.25)
        self.blue_tower_pos_top_c = MakePos(802.81, 4052.36)

        self.red_tower_pos_bot_a = MakePos(13459.0, 4284.0)
        self.red_tower_pos_bot_b = MakePos(12920.0, 8005.0)
        self.red_tower_pos_bot_c = MakePos(13205.0, 10474.0)

        self.red_tower_pos_mid_a = MakePos(8548.0, 8289.0)
        self.red_tower_pos_mid_b = MakePos(9361.0, 9892.0)
        self.red_tower_pos_mid_c = MakePos(10743.0, 11010.0)

        self.red_tower_pos_top_a = MakePos(3911.0, 13654.0)
        self.red_tower_pos_top_b = MakePos(7536.0, 13190.0)
        self.red_tower_pos_top_c = MakePos(10261.0, 13465.0)

        self.blue_tower_pos_nexus_top = MakePos(1271.097, 1989.8077)
        self.blue_tower_pos_nexus_bot = MakePos(1821.097, 1589.8077)

        self.red_tower_pos_nexus_top = MakePos(12621.097, 12364.8077)
        self.red_tower_pos_nexus_bot = MakePos(12171.097, 12789.8077)

        self.blue_inhib_pos_top = MakePos(796.097, 3339.8077)
        self.blue_inhib_pos_mid = MakePos(2746.097, 2964.8077)
        self.blue_inhib_pos_bot = MakePos(2996.097, 1014.8077)

        self.red_inhib_pos_top = MakePos(10946.097, 13414.8077)
        self.red_inhib_pos_mid = MakePos(11196.097, 11439.8077)
        self.red_inhib_pos_bot = MakePos(13196.097, 11164.8077)

        self.blue_nexus_pos = MakePos(1146.097, 1414.8077)
        self.red_nexus_pos = MakePos(12771.097, 13014.8077)

        # other
        self.blue_gromp_pos = MakePos(1950, 7960)
        self.blue_wolves_pos = MakePos(3600, 6000)
        self.blue_raptors_pos = MakePos(6800, 4900)
        self.blue_krugs_pos = MakePos(8100, 2100)

        self.red_gromp_pos = MakePos(12300, 6000)
        self.red_wolves_pos = MakePos(10800, 7900)
        self.red_raptors_pos = MakePos(7600, 9100)
        self.red_krugs_pos = MakePos(6200, 11800)

        self.small_camps = [(None, 'None')]
        for m, label in self.miscs:
            if 'lesser_jungle_icon_v2' == label or 'camp_respawn_urgent' == label:
                self.small_camps.append((m, label))

        self.blue_blue_buff_pos = MakePos(3500, 7600)
        self.blue_red_buff_pos = MakePos(7500, 3500)

        self.red_blue_buff_pos = MakePos(10800, 6500)
        self.red_red_buff_pos = MakePos(6800, 10500)

        self.buff_camps = [(None, 'None')]
        for m, label in self.miscs:
            if 'camp' == label or 'camp_respawn' == label or 'camp_respawn_urgent' == label:
                self.buff_camps.append((m, label))

        self.top_scuttle = MakePos(4200, 9200)
        self.bot_scuttle = MakePos(10200, 4700)

        self.scuttle = [(None, 'None')]
        for m, label in self.miscs:
            if 'lesser_jungle_icon_v2' == label or 'camp_respawn' == label or 'camp_respawn_urgent' == label or 'minimap_ward_green' == label or 'minimap_ward_green_enemy' == label:
                self.scuttle.append((m, label))

        self.blue_gromp_plant_pos = MakePos(2900, 9200)
        self.blue_gromp_plant_vision_pos = MakePos(3300, 8200)
        self.blue_gromp_plant_heal_pos = MakePos(3500, 9250)
        self.blue_gromp_plant_heal2_pos = MakePos(4200, 8700)
        self.blue_blue_buff_plant_pos = MakePos(4400, 6600)

        self.blue_red_buff_base_plant_pos = MakePos(6000, 2800)
        self.blue_red_buff_plant_pos = MakePos(8800, 3100)
        self.blue_red_buff_plant_vision_pos = MakePos(8700, 4000)

        self.red_gromp_plant_pos = MakePos(11700, 4400)
        self.red_gromp_plant_vision_pos = MakePos(11200, 5400)
        self.red_gromp_plant_heal_pos = MakePos(11400, 4200)
        self.red_blue_buff_plant_pos = MakePos(10200, 7000)

        self.red_red_buff_base_plant_pos = MakePos(8600, 10700)
        self.red_red_buff_plant_pos = MakePos(5950, 10400)
        self.red_red_buff_plant_vision_pos = MakePos(5930, 9300)

        self.plants = [(None, 'None')]
        for m, label in self.miscs:
            if 'plant_icon_green' == label:
                self.plants.append((m, label))

        self.baron_pos = MakePos(4700, 9900)
        self.dragon_pos = MakePos(9700, 4000)

        self.baron = [(None, 'None')]
        for m, label in self.miscs:
            if 'sru_riftherald_minimap_icon' == label or 'baron_minimap_icon' == label:
                self.baron.append((m, label))

        self.dragon = [(None, 'None')]
        for m, label in self.miscs:
            if 'dragonairminimap' == label or 'dragonearthminimap' == label or 'dragonfireminimap' == label or 'dragonwaterminimap' == label or 'dragonelderminimap' == label:
                self.dragon.append((m, label))

        self.buff_camps = [(None, 'None')]
        self.ally_champion_outlines = []
        self.enemy_champion_outlines = []
        for m, label in self.miscs:
            m = m.numpy()
            m = m.transpose((2, 1, 0))
            m = cv2.resize(m, (120, 120), interpolation=cv2.INTER_AREA)
            m = m.transpose((2, 1, 0))
            m = torch.from_numpy(m)

            if 'ally_circle' == label or 'recalloutline' == label or 'teleporthighlight_friendly' == label or 'teleporthighlight_shen' == label:
                self.ally_champion_outlines.append((m, label))
            if 'enemy_circle' == label or 'recallhostileoutline' == label or 'teleporthighlight_enemy' == label or 'teleporthighlight_shen' == label:
                self.enemy_champion_outlines.append((m, label))

        self.blue_minion = None
        self.red_minion = None
        self.pings = []
        self.wards = []
        self.shop = None
        self.blue_shop_pos = MakePos(0, 800)
        self.red_shop_pos = MakePos(14200, 14200)
        for m, label in self.miscs:
            if 'minionmapcircle_ally' == label:
                self.blue_minion = (m, label)
            elif 'minionmapcircle_enemy' == label:
                self.red_minion = (m, label)
            elif 'pingcomehere' == label or 'pinggetback' == label or 'pingmia' == label or 'pingomw' == label:
                m = m.numpy()
                m = m.transpose((2, 1, 0))
                m = cv2.resize(m, (32, 32), interpolation=cv2.INTER_AREA)
                m = m.transpose((2, 1, 0))
                m = torch.from_numpy(m)
                self.pings.append((m, label))
            elif 'pingmarker' == label or 'pingmarker_green' == label or 'pingmarker_red' == label:
                self.pings.append((m, label))
            elif 'minimap_jammer' in label or 'minimap_ward' in label:
                self.wards.append((m, label))
            elif 'shop' == label:
                self.shop = (m, label)

    #################################################
    # Helpers
    #################################################

    def sample_list(self, l, amount):
        """Samples amount from list

        Args:
            l ([]): list of objects
            amount (int): amount to sample

        Returns:
            [] : list of objects that are sampled
        """
        vals = np.random.choice(len(l), size=amount, replace=False)
        samples = [l[i] for i in vals]
        return samples

    def overlay_transparent(self, background, overlay, x, y, overlay_size=None):
        """
        @brief      Overlays a transparant PNG onto another image using CV2

        @param      background_img    The background image
        @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
        @param      x                 x location to place the top-left corner of our overlay
        @param      y                 y location to place the top-left corner of our overlay
        @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

        @return     Background image with overlay on top
        """

        background_width = background.shape[1]
        background_height = background.shape[0]

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]
        if h == 0 or w == 0:
            return

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
                ],
                axis=2,
            )

        overlay_image = overlay[..., :4]
        mask = overlay[..., 3:]

        #print(background.shape, overlay.shape, overlay_image.shape, y, y+h, x, x+h)
        x = max(x, 0)
        y = max(y, 0)
        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background

    def overlay(self, large_image, small_image, y=0, x=0):
        """Overlays large image on top of small_image

        Args:
            large_image (np.array): larger image
            small_image (np.array): smaller image to be overlaied on top of large image
            y (int, optional): x location of smaller image placement on top of large image. Defaults to 0.
            x (int, optional): y location of smaller image placement on top of large image. Defaults to 0.

        Returns:
            np.array: overlaid image
        """
        small_image = small_image.transpose((2, 1, 0))

        large_image = large_image.transpose((2, 1, 0))

        self.overlay_transparent(large_image, small_image, x, y)

        large_image = large_image.transpose((2, 1, 0))
        return large_image

    #################################################
    # Generation
    #################################################

    def create_minimap(self):
        minimap, _ = random.choice(self.minimaps)
        minimap = minimap.numpy().copy()
        fog, _ = random.choice(self.fogs)
        fog = fog.numpy()

        # mix fog and minimap
        alpha = 0.9
        minimap = cv2.addWeighted(minimap, alpha, fog, 1 - alpha, 0)

        return minimap

    def create_champions(self, champion_icons, champion_outlines):
        champions = []
        for c, label in champion_icons:
            c = c.numpy().copy()

            # high probably choosing default circle
            consider_other_than_outline = 0
            while consider_other_than_outline != 10:
                circle, c_label = self.sample_list(champion_outlines, 1)[0]
                if 'circle' in c_label:
                    break
                else:
                    consider_other_than_outline += 1

            circle = circle.numpy()
            c = self.overlay(c, circle)
            champions.append((c, label))

        return champions

    def create_ally_and_enemy_champions(self):
        champion_icons = self.sample_list(self.champion_icons, 10)
        ally_champion_icons = champion_icons[:5]
        enemy_champion_icons = champion_icons[5:]

        ally_champions = self.create_champions(ally_champion_icons, self.ally_champion_outlines)
        enemy_champions = self.create_champions(enemy_champion_icons, self.enemy_champion_outlines)

        return (ally_champions, enemy_champions)

    def draw_towers(self, minimap, DrawOperation, FogFilterOperation):
        # towers
        def load_tower_internal(minimap, img, pos, randomness):
            if random.random() < randomness:
                return minimap

            img = img.numpy()
            x, y = pos
            h, w = minimap.shape[1:3]
            x = int(x * w)
            y = int(y * h)

            DrawOperation(img, x, y)
            FogFilterOperation(img, x, y)

        def load_tower(minimap, l, pos, randomness=0.1):
            if isinstance(l, list):
                tower, _ = random.choice(l)
            else:
                tower, _ = l

            load_tower_internal(minimap, tower, pos, randomness)

        load_tower(minimap, self.red_outer_towers, self.red_tower_pos_top_a)
        load_tower(minimap, self.red_outer_towers, self.red_tower_pos_mid_a)
        load_tower(minimap, self.red_outer_towers, self.red_tower_pos_bot_a)

        load_tower(minimap, self.blue_outer_towers, self.blue_tower_pos_top_a)
        load_tower(minimap, self.blue_outer_towers, self.blue_tower_pos_mid_a)
        load_tower(minimap, self.blue_outer_towers, self.blue_tower_pos_bot_a)

        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_top_b)
        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_mid_b)
        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_bot_b)

        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_top_b)
        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_mid_b)
        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_bot_b)

        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_top_c)
        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_mid_c)
        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_bot_c)

        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_top_c)
        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_mid_c)
        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_bot_c)

        load_tower(minimap, self.blue_inhib, self.blue_inhib_pos_top)
        load_tower(minimap, self.blue_inhib, self.blue_inhib_pos_mid)
        load_tower(minimap, self.blue_inhib, self.blue_inhib_pos_bot)

        load_tower(minimap, self.red_inhib, self.red_inhib_pos_top)
        load_tower(minimap, self.red_inhib, self.red_inhib_pos_mid)
        load_tower(minimap, self.red_inhib, self.red_inhib_pos_bot)

        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_nexus_top)
        load_tower(minimap, self.blue_normal_tower, self.blue_tower_pos_nexus_bot)

        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_nexus_top)
        load_tower(minimap, self.red_normal_tower, self.red_tower_pos_nexus_bot)

        load_tower(minimap, self.blue_nexus, self.blue_nexus_pos)
        load_tower(minimap, self.red_nexus, self.red_nexus_pos)

        load_tower(minimap, self.shop, self.blue_shop_pos, 0.0)
        load_tower(minimap, self.shop, self.red_shop_pos, 0.0)

    def draw_neutral_camps_and_plants(self, minimap, DrawOperation):
        def load_maybe_spawn(minimap, l, pos):
            img, _ = random.choice(l)
            if img == None:
                return minimap
            img = img.numpy()
            x, y = pos
            h, w = minimap.shape[1:3]
            x = int(x * w)
            y = int(y * h)

            DrawOperation(img, x, y)

        load_maybe_spawn(minimap, self.small_camps, self.blue_gromp_pos)
        load_maybe_spawn(minimap, self.small_camps, self.blue_wolves_pos)
        load_maybe_spawn(minimap, self.small_camps, self.blue_raptors_pos)
        load_maybe_spawn(minimap, self.small_camps, self.blue_krugs_pos)
        load_maybe_spawn(minimap, self.small_camps, self.red_gromp_pos)
        load_maybe_spawn(minimap, self.small_camps, self.red_wolves_pos)
        load_maybe_spawn(minimap, self.small_camps, self.red_raptors_pos)
        load_maybe_spawn(minimap, self.small_camps, self.red_krugs_pos)

        load_maybe_spawn(minimap, self.buff_camps, self.blue_blue_buff_pos)
        load_maybe_spawn(minimap, self.buff_camps, self.blue_red_buff_pos)
        load_maybe_spawn(minimap, self.buff_camps, self.red_blue_buff_pos)
        load_maybe_spawn(minimap, self.buff_camps, self.red_red_buff_pos)

        load_maybe_spawn(minimap, self.scuttle, self.top_scuttle)
        load_maybe_spawn(minimap, self.scuttle, self.bot_scuttle)

        load_maybe_spawn(minimap, self.plants, self.blue_gromp_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_gromp_plant_vision_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_gromp_plant_heal_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_gromp_plant_heal2_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_blue_buff_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_red_buff_base_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_red_buff_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.blue_red_buff_plant_vision_pos)
        load_maybe_spawn(minimap, self.plants, self.red_gromp_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.red_gromp_plant_vision_pos)
        load_maybe_spawn(minimap, self.plants, self.red_gromp_plant_heal_pos)
        load_maybe_spawn(minimap, self.plants, self.red_blue_buff_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.red_red_buff_base_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.red_red_buff_plant_pos)
        load_maybe_spawn(minimap, self.plants, self.red_red_buff_plant_vision_pos)

        load_maybe_spawn(minimap, self.baron, self.baron_pos)
        load_maybe_spawn(minimap, self.dragon, self.dragon_pos)

    def draw_minions(self, minimap, DrawOperation, FogFilterOperation):
        w, h = minimap.shape[1:3]

        b_minion, label = self.blue_minion
        r_minion, label = self.red_minion
        b_minion = b_minion.numpy()
        r_minion = r_minion.numpy()

        def spawn_minion_side(x, h, minion):
            noise = 50
            diff = 100
            is_line = random.random()
            if is_line < 0.5:
                y = random.randint(diff, h)
                blue_or_red_side = random.random()
                if blue_or_red_side < 0.5:
                    x, y = y, x
                for i in range(6):
                    if blue_or_red_side < 0.5:
                        x += 10
                    else:
                        y += 10
                    DrawOperation(minion, x, y)
                    FogFilterOperation(minion, x, y)
            else:
                # grouped
                y = random.randint(diff, h)
                blue_or_red_side = random.random()
                if blue_or_red_side < 0.5:
                    x, y = y, x
                for i in range(6):
                    x_diff = random.randint(-15, 15)
                    y_diff = random.randint(-15, 15)
                    x_new = x + x_diff
                    y_new = y + y_diff
                    DrawOperation(minion, x_new, y_new)
                    FogFilterOperation(minion, x_new, y_new)

        def spawn_minion_middle(h, w, minion):
            diff = 100
            x = random.randint(diff, h - diff)
            y = h - x

            is_line = random.random()
            if is_line < 0.5:
                for i in range(6):
                    x += 13
                    y -= 13
                    DrawOperation(minion, x, y)
                    FogFilterOperation(minion, x, y)
            else:
                # grouped
                blue_or_red_side = random.random()
                for i in range(6):
                    x_diff = random.randint(-15, 15)
                    y_diff = random.randint(-15, 15)
                    x_new = x + x_diff
                    y_new = y + y_diff
                    DrawOperation(minion, x_new, y_new)
                    FogFilterOperation(minion, x_new, y_new)

        spawn_minion_side(45, h, b_minion)
        spawn_minion_side(45, h, r_minion)
        spawn_minion_side(w - 45, h, b_minion)
        spawn_minion_side(w - 45, h, r_minion)

        spawn_minion_middle(w, h, b_minion)
        spawn_minion_middle(w, h, r_minion)

    def draw_wards(self, minimap, DrawOperation, FogFilterOperation):
        w, h = minimap.shape[1:3]

        # wards
        num_wards = random.randint(0, 10)
        for i in range(num_wards):
            ward, label = self.sample_list(self.wards, 1)[0]
            ward = ward.numpy()
            constraint_distance = 50
            x = random.randint(constraint_distance, w - constraint_distance)
            y = random.randint(constraint_distance, h - constraint_distance)
            DrawOperation(ward, x, y)
            FogFilterOperation(ward, x, y)

    def draw_champions(self, minimap, DrawOperation, FogFilterOperation, ally_champions, enemy_champions):
        champion_position_data = []

        def AddChampions(minimap, champions):
            w, h = minimap.shape[1:3]
            for c, label in champions:
                c_w, c_h = c.shape[1:3]

                c = c.transpose((2, 1, 0))
                c = cv2.resize(c, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                c = c.transpose((2, 1, 0))
                c_w, c_h = c.shape[1:3]
                c_w = int(c_w / 2.0)
                c_h = int(c_h / 2.0)
                x = random.randint(-c_w + 1, w - c_w)
                y = random.randint(-c_h + 1, h - c_h)

                if len(champion_position_data) > 0:
                    if random.random() < 0.2:
                        other_champion = random.choice(champion_position_data)
                        other_c, other_x, other_y, other_c_w, _ = other_champion
                        #print(other_c, other_x, other_y)
                        nearness = c_w
                        bias_x = pow(random.random(), 0.5) * (-1 if random.random() < 0.5 else 1)
                        bias_y = pow(random.random(), 0.5) * (-1 if random.random() < 0.5 else 1)
                        x = int(other_x + nearness * bias_x)
                        y = int(other_y + nearness * bias_y)
                        #print(other_c, other_x, other_y, label, x, y)
                        x = max(-c_w + 1, min(x, w - c_w))
                        y = max(-c_h + 1, min(y, h - c_h))

                orig_x = x
                orig_y = y

                if x >= w - c_w * 2.0:
                    c = c[:, :, :int(w-x)]
                if x < 0:
                    c = c[:, :, -int(c_w * 2 + x):]
                    x = 0
                if y >= h - c_h * 2.0:
                    c = c[:, :int(h-y), :]
                if y < 0:
                    c = c[:, -int(c_h * 2 + y):, :]
                    y = 0

                c_w, c_h = c.shape[1:3]

                state = (label, int(y * self.minimap_ratio[1]), int(self.minimap_ratio[0] * x), int(self.minimap_ratio[1] * c_w), int(self.minimap_ratio[0] * c_h))

                DrawOperation(c, x, y)
                FogFilterOperation(c, x, y)

                champion_position_data.append(state)

        AddChampions(minimap, ally_champions)
        AddChampions(minimap, enemy_champions)

        return champion_position_data

    def draw_pings(self, minimap, DrawOperation, FogFilterOperation, champion_position_data=None):
        w, h = minimap.shape[1:3]
        num_pings = random.randint(3, 10)
        for i in range(num_pings):
            ping, label = self.sample_list(self.pings, 1)[0]
            ping = ping.numpy()
            constraint_distance = 0  # distance away from center
            x = random.randint(constraint_distance, w - constraint_distance)
            y = random.randint(constraint_distance, h - constraint_distance)

            if random.random() < 0.9 and champion_position_data is not None:
                other_champion = random.choice(champion_position_data)
                other_c, other_x, other_y, other_c_w, other_c_h = other_champion
                other_x *= 1 / self.minimap_ratio[0]
                other_y *= 1 / self.minimap_ratio[1]
                nearness = other_c_w
                #bias_x = pow(random.random(), 0.5) * (-1 if random.random() < 0.5 else 1)
                #bias_y = pow(random.random(), 0.5) * (-1 if random.random() < 0.5 else 1)
                #x = int(other_x + nearness * bias_x)
                #y = int(other_y + nearness * bias_y)
                #x = max(-constraint_distance + 1, min(x, w - constraint_distance))
                #y = max(-constraint_distance + 1, min(y, h - constraint_distance))
                x = other_y
                y = other_x
                bias_x = random.randint(1, 1) * random.random() * (-1 if random.random() < 0.5 else 1)
                bias_y = random.randint(1, 1) * random.random() * (-1 if random.random() < 0.5 else 1)
                x = int(x + nearness * bias_x)
                y = int(y + nearness * bias_y)

            x = int(x)
            y = int(y)

            # create circle wave
            ping_h, ping_w = ping.shape[1:3]
            ping = ping.transpose((2, 1, 0))

            color = ping[int(ping_h / 2), int(ping_w / 2), :]
            color = [x.item() for x in color]

            ww = random.randint(ping_w, ping_w * 5)
            ww_h = int(ww / 2)

            circle_x = x - ww_h + int(ping_w)  # int(ping_w / 2)
            circle_y = y - ww_h + int(ping_h)  # int(ping_h / 2)
            circle = np.zeros((ww, ww, 4), dtype=np.float32)
            ping_center_x, ping_center_y = ww_h, ww_h

            number_times = random.randint(0, 4)
            for i in range(number_times):
                radius = random.randint(1, ww_h)
                thickness = random.randint(1, ww_h)
                color[3] = min(0.6, max(0.1, random.random()))
                if thickness < 20:
                    color[3] *= min(0.8, color[3] * 1.5)
                circle = cv2.circle(circle, (ping_center_x, ping_center_y), radius, color, thickness, lineType=cv2.LINE_AA)

            circle = circle.transpose((2, 1, 0))

            ping = ping.transpose((2, 1, 0))

            DrawOperation(circle, circle_x, circle_y)
            DrawOperation(ping, x, y)

    #################################################
    # Final modifications
    #################################################

    def perform_fog_operations(self, minimap, fog_filter_operations):
        fog_filter = np.zeros(minimap.shape, minimap.dtype)
        fog_filter = fog_filter.transpose((2, 1, 0)).copy()

        for img, x, y in fog_filter_operations:
            h, w = img.shape[1:3]
            distance = 40

            x = int(x + w / 2)
            y = int(y + h / 2)

            fog_filter = cv2.circle(fog_filter, (x, y), distance, (0.0, 0.0, 0.0, 1.0), -1)

        fog_filter_black = fog_filter
        fog_filter = fog_filter.transpose((2, 1, 0))

        fog_mask = fog_filter.astype('uint8')
        fog_mask = fog_mask.transpose((2, 1, 0))
        fog_mask[fog_mask[...] == 1] = 255
        fog_mask = cv2.split(fog_mask)[3]

        fog_filter = fog_filter.transpose((2, 1, 0))
        minimap = minimap.transpose((2, 1, 0))

        fog_filter = cv2.bitwise_or(minimap, fog_filter, mask=fog_mask)
        fog_filter[fog_filter[..., 3] == 0] = (0.0, 0.0, 0.0, 1.0)

        minimap = minimap.transpose((2, 1, 0))
        fog_filter = fog_filter.transpose((2, 1, 0))

        alpha = 0.3
        minimap = cv2.addWeighted(minimap, alpha, fog_filter, 1 - alpha, 0)

        return minimap

    def perform_overlay_operations(self, minimap, overlay_operations):
        for img, x, y in overlay_operations:
            minimap = self.overlay(minimap, img, x, y)

        return minimap

    def draw_lines_and_boxes(self, minimap, champion_position_data):
        w, h = minimap.shape[1:3]

        minimap = minimap.transpose((2, 1, 0)).copy()

        # draw random line
        label, x, y, c_w, c_h = self.sample_list(champion_position_data, 1)[0]
        my_champion_point = [int(x + (c_w / 2)), int(y + (c_h / 2))]
        target_random_point = [random.randint(0, w), random.randint(0, h)]
        points = []

        diff_x = (target_random_point[0] - my_champion_point[0])
        diff_y = (target_random_point[1] - my_champion_point[1])
        for i in range(5):
            i /= 5.0
            noise_size_x = int(diff_x / 5)
            noise_size_y = int(diff_y / 5)
            x_noise_begin = min(noise_size_x, 0)
            x_noise_end = max(noise_size_x, 0)
            y_noise_begin = min(noise_size_y, 0)
            y_noise_end = max(noise_size_y, 0)

            noise_x = random.randint(x_noise_begin, x_noise_end)
            noise_y = random.randint(y_noise_begin, y_noise_end)
            p_x = int((my_champion_point[0] + (diff_x * i) + noise_x) * (1 / self.minimap_ratio[0]))
            p_y = int((my_champion_point[1] + (diff_y * i) + noise_y) * (1 / self.minimap_ratio[1]))
            p = (p_x, p_y)
            points.append(p)

        for point1, point2 in zip(points, points[1:]):
            cv2.line(minimap, point1, point2, (1.0, 1.0, 1.0, 1.0), 2)

        h, w = minimap.shape[1:3]

        def draw_rectangle(minimap, x_offset=None, y_offset=None):
            white_rectangle_height = random.randint(80, 120)
            white_rectangle_width = random.randint(160, 240)
            white_rectangle_width_2 = int(white_rectangle_width / 2)
            white_rectangle_height_2 = int(white_rectangle_height / 2)
            target_random_point = [random.randint(-white_rectangle_width_2, w + white_rectangle_width_2), random.randint(-white_rectangle_height_2, h + white_rectangle_height_2)]
            t_y, t_x = target_random_point

            if y_offset is not None:
                diff_y = random.randint(-white_rectangle_height, int(white_rectangle_height / 2))
                t_y = y_offset + diff_y
                diff_x = random.randint(-white_rectangle_width, int(white_rectangle_width / 2))
                t_x = x_offset + diff_x

            minimap = cv2.rectangle(minimap, (t_y, t_x), (t_y + white_rectangle_height, t_x + white_rectangle_width), (1.0, 1.0, 1.0, 1.0), 3)
            return minimap

        minimap = draw_rectangle(minimap)
        rectangles = 5  # random.randint(2, 5)
        for i in range(rectangles):
            if random.random() < 0.25:
                other_champion = random.choice(champion_position_data)
                other_c, other_x, other_y, other_c_w, other_c_h = other_champion
                x_diff = (other_c_w * 3)
                x = other_x
                y = other_y
                if random.random() < 0.5:
                    x = (other_x + other_c_w) / 2
                    x_offset = random.uniform(-1, 1) * other_c_w / 2
                    x += x_offset
                if random.random() < 0.5:
                    y = (other_y + other_c_h) / 2
                    y_offset = random.uniform(-1, 1) * other_c_h / 2
                    y += y_offset

                x = int(x)
                y = int(y)
                minimap = draw_rectangle(minimap, x, y)

        minimap = minimap.transpose((2, 1, 0))
        return minimap

    def perform_final_touches_to_image(self, minimap):
        minimap = minimap.transpose((2, 1, 0))
        minimap = cv2.cvtColor(minimap, cv2.COLOR_RGBA2RGB)
        minimap = minimap.transpose((2, 1, 0))

        # wrap up
        if self.resize is not None:
            minimap = minimap.transpose((2, 1, 0))
            minimap = cv2.resize(minimap, self.resize, interpolation=cv2.INTER_AREA)
            minimap = minimap.transpose((2, 1, 0))

        img = torch.as_tensor(minimap)

        return img

    def convert_champion_position_data_to_coco_format(self, champion_position_data, index):
        # construct boxes
        boxes = []
        labels = []
        area = []
        image_id = index
        iscrowd = []
        for label, x, y, c_w, c_h in champion_position_data:
            xmin = float(x)
            xmax = float(x + c_w)
            ymin = float(y)
            ymax = float(y + c_h)
            entry = [ymin, xmin, ymax, xmax]
            #y1, x1, y2, x2

            boxes.append(entry)

            champion_id = self.champion_to_id[label]
            labels.append(champion_id)

            area_ = c_w * c_h
            area.append(area_)

            iscrowd.append(0)  # 1 if ignore

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([image_id], dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

        data = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
        }

        return data

    #################################################
    # Main call
    #################################################

    # obtain the sample with the given index
    def generate_data(self, index):
        minimap = self.create_minimap()

        # overlay_operations refers to overlay of images on top of minimap
        # fog_filter_operations refers to highlighting areas of minimap due to visibility
        overlay_operations = []
        fog_filter_operations = []

        def DrawOperation(img, x, y):
            overlay_operations.append((img, x, y))

        def FogFilterOperation(img, y, x):
            fog_filter_operations.append((img, x, y))

        self.draw_towers(minimap, DrawOperation, FogFilterOperation)
        self.draw_neutral_camps_and_plants(minimap, DrawOperation)
        self.draw_minions(minimap, DrawOperation, FogFilterOperation)
        self.draw_wards(minimap, DrawOperation, FogFilterOperation)

        # load champions
        ally_champions, enemy_champions = self.create_ally_and_enemy_champions()
        champion_position_data = self.draw_champions(minimap, DrawOperation, FogFilterOperation, ally_champions, enemy_champions)

        self.draw_pings(minimap, DrawOperation, FogFilterOperation, champion_position_data=champion_position_data)

        # add fog light
        minimap = self.perform_fog_operations(minimap, fog_filter_operations)
        minimap = self.perform_overlay_operations(minimap, overlay_operations)

        minimap = self.draw_lines_and_boxes(minimap, champion_position_data)

        final_image = self.perform_final_touches_to_image(minimap)
        data = self.convert_champion_position_data_to_coco_format(champion_position_data, index)

        return (final_image, data)
