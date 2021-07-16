# -*- coding: utf-8 -*-
import socket
 
#创建一个socket对象
def receive(self):
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM,0)
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    #sock.setblocking(1)
    ip_port = ('127.0.0.1',9998)
    #绑定ip和端口号
    sock.bind(ip_port)
    #sock.setblocking(1)
    #设置最大连接数
    sock.listen(5)
    
    
    while True:
        #使用accept方法获取一个客户端连接
        #获取客户端的scoket对象conn和客户端的地址(ip、端口号)address
        conn,address = sock.accept()
        # 给客户端发信息
        send_data = 'Hello.'
        send_data = send_data.encode()
        conn.sendall(send_data)
        recv_data_ = []
        return_data = []
        flag = False
        while True:
            try:
                # 接收客户端消息
                recv_data = conn.recv(1024)
                #print 'Client:', recv_data, ", Type:", type(recv_data), ", Equal:", (recv_data.replace("0x00", "") == 'start')
                #print 'Client:', recv_data[0:5], ", Type:", type(recv_data[0:5])
                # 如果收到start就开始调用统计代码
                for a in recv_data:
                    if a == ' ':
                        recv_data_.append(chr(a))
                    elif a < 22:
                        break
                    else:
                        recv_data_.append(a - 48)
                        flag = True
                        
                last = 0
                for a in recv_data_:
                    if a <= -16:
                        return_data.append(last)
                        last = 0
                    else:
                        last = last*10
                        last += int(a)
                return_data.append(last)
                print(return_data[0:2])
                if flag:
                    # break
                    return return_date

            except Exception as e:
                break
    
        # 关闭客户端的socket连接
        conn.close()