import argparse
from PIL import Image
import os

def resize_images(input_dir, target_size=(64, 64, 3), progress_interval=100, count=0):
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # 检查是否为图片文件
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # 打开图片
                img = Image.open(input_path)

                # 调整尺寸
                resized_img = img.resize(target_size[:2])

                # 覆盖原始图片
                resized_img.save(input_path)
                
                count += 1

                # 每隔1张图片打印进度
                print(f'Processed {count} images')

            except Exception as e:
                print(f'Error processing {input_path}: {e}')
        elif os.path.isdir(input_path):
            # 递归调用处理子目录
            count = resize_images(input_path, target_size, progress_interval, count)

    return count

def main():
    parser = argparse.ArgumentParser(description='Resize images in a directory in-place.')
    parser.add_argument('input_directory', help='Input directory containing images to be resized')
    args = parser.parse_args()

    # 调整图片尺寸
    total_count = resize_images(args.input_directory)

    print(f'Total {total_count} images processed.')

if __name__ == "__main__":
    main()
