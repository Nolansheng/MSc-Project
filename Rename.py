import os


class rename():
    def __init__(self, path, label_path):
        self.path = path
        self.label_path = label_path

    def train_rename(self):
        train_list = os.listdir(self.path)
        train_label_list = os.listdir(self.label_path)
        i = 1
        for train_item in train_list:
            item = train_item
            train_item = train_item.split('.')

            for train_item_label in train_label_list:
                src = train_item[0] + '_L'
                label_item = train_item_label
                src_label = train_item_label.split('.')
                src_label = src_label[0]

                if src == src_label:
                    src1 = os.path.join(os.path.abspath(self.path), item)
                    dst1 = os.path.join(os.path.abspath(self.path), 'train_' + str(i) + '.png')

                    src2 = os.path.join(os.path.abspath(self.label_path), label_item)
                    dst2 = os.path.join(os.path.abspath(self.label_path), 'train_label_' + str(i) + '.png')

                    try:
                        os.rename(src1, dst1)
                        os.rename(src2, dst2)
                        i += 1

                    except:
                        continue
        print('rename successful')

    def test_rename(self):
        test_list = os.listdir(self.path)
        test_label_list = os.listdir(self.label_path)

        i = 1
        for test_item in test_list:
            item = test_item
            test_item = test_item.split('.')

            for test_item_label in test_label_list:
                src = test_item[0] + '_L'
                label_item = test_item_label
                src_label = test_item_label.split('.')
                src_label = src_label[0]

                if src == src_label:
                    src1 = os.path.join(os.path.abspath(self.path), item)
                    dst1 = os.path.join(os.path.abspath(self.path), 'test_' + str(i) + '.png')

                    src2 = os.path.join(os.path.abspath(self.label_path), label_item)
                    dst2 = os.path.join(os.path.abspath(self.label_path), 'test_label_' + str(i) + '.png')

                    try:
                        os.rename(src1, dst1)
                        os.rename(src2, dst2)
                        i += 1

                    except:
                        continue
        print('rename successful')

    def val_rename(self):
        val_list = os.listdir(self.path)
        val_label_list = os.listdir(self.label_path)
        i = 1
        for val_item in val_list:
            item = val_item
            val_item = val_item.split('.')

            for val_item_label in val_label_list:
                src = val_item[0] + '_L'
                label_item = val_item_label
                src_label = val_item_label.split('.')
                src_label = src_label[0]

                if src == src_label:
                    src1 = os.path.join(os.path.abspath(self.path), item)
                    dst1 = os.path.join(os.path.abspath(self.path), 'val_' + str(i) + '.png')

                    src2 = os.path.join(os.path.abspath(self.label_path), label_item)
                    dst2 = os.path.join(os.path.abspath(self.label_path), 'val_label_' + str(i) + '.png')

                    try:
                        os.rename(src1, dst1)
                        os.rename(src2, dst2)
                        i += 1

                    except:
                        continue
        print('rename successful')