from torch.utils import data
import multiprocessing
import queue
import threading

from image_drawer import ImageDrawer


class ImageGeneratorDataset(data.Dataset):
    def __init__(self, champion_circle_icon_path, minimap_path,
                 fog_path, misc_path, dataset_length,
                 save_mode=False, resize=None, thread_pool_size=1):
        self.champion_circle_icon_path = champion_circle_icon_path
        self.minimap_path = minimap_path
        self.fog_path = fog_path
        self.misc_path = misc_path
        self.dataset_length = dataset_length
        self.save_mode = save_mode
        self.resize = resize
        self.thread_pool_size = thread_pool_size

        self.image_drawer = ImageDrawer(self.champion_circle_icon_path,
                                        self.minimap_path, self.fog_path, self.misc_path,
                                        self.resize)

        self.datas = queue.Queue(self.dataset_length)
        if not self.save_mode:
            if self.thread_pool_size == 1:
                for i in range(dataset_length):
                    data = self.image_drawer.generate_data(i)
                    self.datas.put(data)
            else:
                self.threads = []
                for i in range(self.thread_pool_size):
                    t = multiprocessing.Process(target=self.thread_generate_data)
                    #t = threading.Thread(target=self.thread_generate_data)
                    t.start()
                    self.threads.append(t)

    def __getitem__(self, index):
        data = self.datas.get()
        self.datas.task_done()
        return data

    def thread_generate_data(self):
        while True:
            data = self.image_drawer.generate_data(0)
            self.datas.put(data)

    def save_to_disk(self, yolo_images_path, yolo_labels_path):
        """Saves to disk a yolo format of images using threading

        Args:
            yolo_images_path (string): folder to write yolo images to
            yolo_labels_path (string): folder to write yolo labels to
        """
        if self.save_mode:
            def Gen(indices):
                for i in indices:
                    data = self.generate_data(i)
                    img, label = data

                    yolo_image_path = os.path.join(yolo_images_path, str(i) + '.jpg')
                    yolo_label_path = os.path.join(yolo_labels_path, str(i) + '.txt')

                    torchvision.utils.save_image(img, yolo_image_path)

                    m_h, m_w = self.minimap_size

                    with open(yolo_label_path, 'w+') as f:
                        ls = label['labels'].numpy().tolist()
                        vs = label['boxes'].numpy().tolist()
                        for l, v in zip(ls, vs):
                            label = l
                            x1, y1, x2, y2 = v
                            center_x = ((x1 + x2) / 2) / m_w
                            center_y = ((y1 + y2) / 2) / m_h
                            w = (x2 - x1) / m_w
                            h = (y2 - y1) / m_h
                            print(x1, y1, x2, y2, m_w, m_h)
                            output = '{} {} {} {} {}\n'.format(label, center_x, center_y, w, h)
                            f.write(output)

            vals = list(range(self.dataset_length))
            vals = np.array_split(vals, self.thread_pool_size)
            vals = [v.tolist() for v in vals]
            p = multiprocessing.pool.ThreadPool(self.thread_pool_size)
            #p = multiprocessing.Pool(self.thread_pool_size)
            p.map(Gen, vals)
            p.close()

    # the total number of samples (optional)
    def __len__(self):
        return self.dataset_length
