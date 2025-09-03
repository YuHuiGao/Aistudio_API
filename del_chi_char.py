import os
import re
import shutil

def has_chinese_chars(s):
    """检查字符串是否包含中文字符"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(s))

def delete_chinese_named_items(root_dir):
    """遍历文件夹，删除名称包含中文的文件和文件夹"""
    if not os.path.exists(root_dir):
        print(f"错误：路径 '{root_dir}' 不存在")
        return
    
    # 先收集所有需要删除的项目，避免边遍历边删除导致的问题
    to_delete = []
    
    # 遍历目录
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 检查文件
        for filename in filenames:
            if has_chinese_chars(filename):
                file_path = os.path.join(dirpath, filename)
                to_delete.append(('file', file_path))
        
        # 检查文件夹
        for dirname in dirnames:
            if has_chinese_chars(dirname):
                dir_path = os.path.join(dirpath, dirname)
                to_delete.append(('dir', dir_path))
    
    if not to_delete:
        print("没有找到名称包含中文的文件或文件夹")
        return
    
    # 显示将要删除的项目
    print("将要删除以下项目：")
    for item_type, path in to_delete:
        print(f"{'文件夹' if item_type == 'dir' else '文件'}: {path}")
    
    # 确认删除
    confirm = input("确定要删除以上所有项目吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消删除操作")
        return
    
    # 执行删除
    deleted_count = 0
    for item_type, path in to_delete:
        try:
            if item_type == 'file':
                os.remove(path)
            else:
                shutil.rmtree(path)
            deleted_count += 1
            print(f"已删除: {path}")
        except Exception as e:
            print(f"删除失败 {path}: {str(e)}")
    
    print(f"操作完成，共删除 {deleted_count} 个项目")

if __name__ == "__main__":
    # 在这里指定要处理的根目录
    target_directory = "./PaddleMIX-develop"  # 替换为实际的文件夹路径
    
    # 确保路径是绝对路径
    target_directory = os.path.abspath(target_directory)
    print(f"开始处理目录: {target_directory}")
    
    delete_chinese_named_items(target_directory)
