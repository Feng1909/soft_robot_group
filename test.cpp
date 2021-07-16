#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <memory.h>
#include <signal.h>
#include <time.h>
#include <cstdlib>
#include <arpa/inet.h>

 
int main() {
 
 
    /*步骤1:创建socket*/
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
 
    if (sockfd < 0) {
        perror("socket error");
        exit(1);
    }
 
 
    struct sockaddr_in serveraddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_port = htons(atoi("9998"));
    inet_pton(AF_INET, "127.0.0.1", &serveraddr.sin_addr.s_addr);
 
    /*步骤2:客户端调用connect函数连接到服务器*/
    if (connect(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
        perror("connect error");    
        exit(1);
    }
 
    /*步骤3:调用IO函数(read/write)进行服务端的双向通信*/
    char buffer[10];
    memset(buffer, 0, sizeof(buffer));
    strcpy(buffer, "12 24");
 
    size_t size;
    printf("buffer is %s, sizeof(buffer) is %d\n", buffer, sizeof(buffer));
 
    if (write(sockfd, buffer, sizeof(buffer)) < 0) {
        perror("write error");  
    }
 
    /*步骤4:关闭socket*/
    close(sockfd);
     
    return 0;
}