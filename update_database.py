#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库更新脚本
用于更新数据库表结构，添加用户表和修改检测结果表
"""

import pymysql

def update_database():
    """更新数据库结构"""
    print("开始更新数据库...")
    
    try:
        # 连接到MySQL服务器
        print("1. 连接到MySQL服务器...")
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="123456",
            charset="utf8mb4"
        )
        print("   ✓ 连接成功")
        
        # 检查数据库是否存在
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES LIKE 'cv_results'")
        if not cursor.fetchone():
            print("   数据库cv_results不存在，正在创建...")
            cursor.execute("CREATE DATABASE cv_results CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("   ✓ 数据库创建成功")
        else:
            print("   ✓ 数据库已存在")
        
        # 选择数据库
        conn.select_db("cv_results")
        print("2. 选择数据库cv_results")
        
        # 检查users表是否存在
        cursor.execute("SHOW TABLES LIKE 'users'")
        if not cursor.fetchone():
            print("3. 创建用户表users...")
            cursor.execute("""
            CREATE TABLE users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                email VARCHAR(100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            print("   ✓ 用户表创建成功")
        else:
            print("3. 用户表已存在")
            # 检查是否有必要的字段
            cursor.execute("DESCRIBE users")
            fields = [row[0] for row in cursor.fetchall()]
            print(f"   当前字段: {fields}")
        
        # 检查detection_results表是否有user_id字段
        print("4. 检查detection_results表结构...")
        cursor.execute("DESCRIBE detection_results")
        fields = {row[0]: row for row in cursor.fetchall()}
        print(f"   当前字段: {list(fields.keys())}")
        
        if 'user_id' not in fields:
            print("   添加user_id字段...")
            # 检查外键约束
            cursor.execute("""
            SELECT CONSTRAINT_NAME FROM information_schema.TABLE_CONSTRAINTS
            WHERE TABLE_SCHEMA = 'cv_results'
            AND TABLE_NAME = 'detection_results'
            AND CONSTRAINT_TYPE = 'FOREIGN KEY'
            """)
            fks = cursor.fetchall()
            for fk in fks:
                print(f"   删除外键: {fk[0]}")
                cursor.execute(f"ALTER TABLE detection_results DROP FOREIGN KEY {fk[0]}")
            
            cursor.execute("ALTER TABLE detection_results ADD COLUMN user_id INT NOT NULL AFTER id")
            cursor.execute("""
            ALTER TABLE detection_results
            ADD CONSTRAINT fk_detection_results_users
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            """)
            print("   ✓ user_id字段添加成功")
        else:
            print("   user_id字段已存在")
        
        # 检查并添加缺失的字段
        print("5. 检查并添加缺失字段...")
        required_fields = {
            'detection_results': ['user_id', 'image_name', 'total_defects', 'detection_time', 'window_size', 'overlap_ratio', 'output_path'],
            'defect_details': ['result_id', 'defect_index', 'confidence', 'class_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'center_x', 'center_y']
        }
        
        for table, required in required_fields.items():
            cursor.execute(f"DESCRIBE {table}")
            existing = {row[0]: row for row in cursor.fetchall()}
            
            for field in required:
                if field not in existing:
                    print(f"   {table}表缺少字段{field}，添加中...")
                    if field == 'user_id':
                        cursor.execute("ALTER TABLE detection_results ADD COLUMN user_id INT NOT NULL AFTER id")
                    elif field in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'center_x', 'center_y', 'confidence']:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {field} FLOAT")
                    elif field == 'defect_index':
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {field} INT")
                    elif field == 'class_name':
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {field} VARCHAR(100)")
                    elif field == 'result_id':
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {field} INT")
                    print(f"   ✓ {field}字段添加成功")
                else:
                    print(f"   ✓ {field}字段已存在")
        
        # 创建测试用户
        print("6. 创建测试用户...")
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (username, password, email) VALUES ('admin', '123456', 'admin@example.com')")
            conn.commit()
            print("   ✓ 测试用户创建成功 (用户名: admin, 密码: 123456)")
        else:
            print("   测试用户已存在")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*50)
        print("🎉 数据库更新完成！")
        print("="*50)
        print("\n您现在可以：")
        print("1. 运行 'python pyqt.py' 启动程序")
        print("2. 使用用户名 'admin' 和密码 '123456' 登录")
        print("3. 或点击'注册新用户'创建新账号")
        return True
        
    except pymysql.MySQLError as err:
        print(f"\n✗ 错误：{err}")
        print("\n可能的原因：")
        print("1. MySQL服务未启动")
        print("2. 用户名或密码错误")
        print("3. 网络连接问题")
        return False
    except Exception as e:
        print(f"\n✗ 未知错误：{e}")
        return False

if __name__ == "__main__":
    update_database()
