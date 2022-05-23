#include <iostream>
#include <unistd.h>
#include "nlohmann/json.hpp"
#include <vector>
#include <filesystem>
#include <cstdio>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <spawn.h>

namespace fs = std::filesystem;

void exit_on_alarm(int sig) {
    exit(0);
}

void run_kart_helper(char *envp[], std::string socket_filename) {
    auto cmd = "kart";
    pid_t pid;
    char *argv[] = {(char *) cmd, (char *) "helper", (char *) "--socket", socket_filename.data(),NULL};
    int status;
    status = posix_spawnp(&pid, (char *) cmd, NULL, NULL, argv, envp);
    if (status < 0) {
        std::cout << "Error running kart helper: " << strerror(status) << std::endl;
        exit(1);
    }
}


int main(int argc, char *argv[], char *envp[]) {
    std::vector<std::string> args(&argv[0], &argv[argc]);
    std::map<std::string, std::string> environ{};
    while (*envp) {
        std::string env = std::string(*envp++);
        auto pos = env.find('=');
        environ[env.substr(0, pos)] = env.substr(pos + 1, env.length());
    }

    FILE *fp = std::fopen(fs::current_path().c_str(), "r");
    int NUM_FD = 4;
    int fds[4] = {fileno(stdin), fileno(stdout), fileno(stderr), fileno(fp)};

    auto socket_filename = environ["HOME"] + std::filesystem::path::preferred_separator + ".kart.socket";
    int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, socket_filename.c_str());


    if (connect(socket_fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        // start helper in background and wait
        run_kart_helper(envp, socket_filename);

        int rtc, max_retry = 10;
        while ((rtc = connect(socket_fd, (struct sockaddr *) &addr, sizeof addr)) != 0
               && --max_retry >= 0) {
            usleep(250000);
        }
        if (rtc < 0) {
            std::cout << "Timeout connecting to kart helper" << std::endl;
            return 2;
        }
    }

    struct msghdr msg = {nullptr};

    char buf[CMSG_SPACE(sizeof(int))];
    msg.msg_control = buf;
    msg.msg_controllen = sizeof buf;

    nlohmann::json j;
    j["pid"] = getpid();
    j["argv"] = args;
    j["environ"] = environ;

    struct iovec base = {nullptr};
    base.iov_base = j.dump(-1, ' ', true).data();
    base.iov_len = strlen(j.dump().data());

    msg.msg_iov = &base;
    msg.msg_iovlen = 1;

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;//passing fd
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * NUM_FD); //size of one fd

    int *fdptr = (int *) CMSG_DATA(cmsg);
    memcpy(fdptr, fds, NUM_FD * sizeof(int));

    msg.msg_controllen = cmsg->cmsg_len;

    signal(SIGALRM, exit_on_alarm);

    if (sendmsg(socket_fd, &msg, 0) < 0) {
        std::cout << "Error sending command to kart helper " << strerror(errno) << std::endl;
        return 3;
    };

    sleep(100);

    std::cout << "Timed out, no response from kart helper" << std::endl;
    return 4;
}
