"""
为整个项目提供统一的绝对路径

"""
import os
def get_project_root():
    """
    获取工程所在的根目录
    :return:字符串根目录
    """
    #获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)#__file__代表当前文件
    #得到此文件的文件夹的目录，相当于往上一级跳一层
    current_file_dir = os.path.dirname(current_file)
    #获取工程所在的根目录
    project_root = os.path.dirname(current_file_dir)
    return project_root

def get_abs_path(relative_path):
    abs_path = os.path.join(get_project_root(),relative_path)
    return abs_path

if  __name__ == "__main__":
    print(get_abs_path("config/config.txt"))
