#include "link.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <IOKit/serial/ioss.h>

namespace pmw {

SerialLink::~SerialLink() { close(); }

void SerialLink::open(const std::string& dev, int baud) {
    close();
    // O_NONBLOCK on the open too, or the open itself blocks waiting for carrier detect on
    // a port with no modem lines. O_NOCTTY: a controlling terminal would deliver SIGHUP
    // here on unplug, and the supervisor owns that decision.
    fd_ = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0)
        throw std::runtime_error("open " + dev + ": " + std::strerror(errno));

    termios t{};
    if (tcgetattr(fd_, &t) != 0)
        throw std::runtime_error("tcgetattr: " + std::string(std::strerror(errno)));
    cfmakeraw(&t);                 // 8N1, no echo, no canonical mode, no flow control
    t.c_cflag |= (CLOCAL | CREAD); // CLOCAL: ignore modem status lines
    t.c_cflag &= ~CRTSCTS;
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;             // reads return immediately with whatever is there
    if (tcsetattr(fd_, TCSANOW, &t) != 0)
        throw std::runtime_error("tcsetattr: " + std::string(std::strerror(errno)));

    // 921600 is not a POSIX-standard rate. On Darwin the way to set an arbitrary rate is
    // IOSSIOSPEED, AFTER tcsetattr -- a later tcsetattr would undo it.
    speed_t sp = (speed_t)baud;
    if (ioctl(fd_, IOSSIOSPEED, &sp) == -1)
        throw std::runtime_error("IOSSIOSPEED " + std::to_string(baud) + ": " +
                                 std::strerror(errno));

    // DTR and RTS LOW. On this board they are wired to EN and IO0, so asserting them is a
    // reset into the bootloader; `link.py` sets `dtr = False, rts = False` at open for
    // exactly this reason, and opening the port must not be able to drop the coils.
    int bits = TIOCM_DTR | TIOCM_RTS;
    ioctl(fd_, TIOCMBIC, &bits);

    tcflush(fd_, TCIOFLUSH);
    pending_.clear();
}

void SerialLink::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SerialLink::send_drive(const float amp_pct[4], const float phase_deg[4], float freq_hz,
                            double now_s) {
    if (fd_ < 0) return false;
    drive_frame::Drive d{};
    d.seq = seq_;
    for (int i = 0; i < 4; ++i) { d.amp_pct[i] = amp_pct[i]; d.phase_deg[i] = phase_deg[i]; }
    d.freq_hz = freq_hz;
    uint8_t buf[drive_frame::DRIVE_LEN];
    drive_frame::encode(d, buf);
    ssize_t n = ::write(fd_, buf, drive_frame::DRIVE_LEN);
    if (n != drive_frame::DRIVE_LEN) {
        st_.n_eagain++;
        return false;
    }
    sent_at_[seq_ % RTT_RING] = now_s;
    seq_++;
    st_.n_sent++;
    return true;
}

bool SerialLink::send_line(const std::string& s) {
    if (fd_ < 0) return false;
    std::string out = s + "\n";
    return ::write(fd_, out.data(), out.size()) == (ssize_t)out.size();
}

void SerialLink::send_stop() {
    if (fd_ < 0) return;
    // Three times, each with a leading newline. The newline ends any half-written ASCII
    // line; the gap before the next byte ends any half-received binary frame; three
    // copies survive a burst of corruption. `stop` is idempotent (`safe_off.py` relies on
    // that), so sending it more than once costs nothing.
    const char* s = "\nstop\n";
    for (int i = 0; i < 3; ++i) (void)!::write(fd_, s, 6);
}

void SerialLink::poll(double now_s, std::vector<std::string>* lines) {
    if (fd_ < 0) return;
    uint8_t buf[4096];
    for (;;) {
        ssize_t n = ::read(fd_, buf, sizeof(buf));
        if (n <= 0) break;
        for (ssize_t i = 0; i < n; ++i) {
            uint8_t b = buf[i];
            // The reverse channel is ACK frames (0xA6) mixed with the 2 Hz ASCII
            // telemetry line. Same two-framing rule as the firmware's, mirrored.
            if (b == drive_frame::SYNC_ACK && pending_.empty()) {
                ack_buf_[0] = b;
                ack_n_ = 1;
                continue;
            }
            if (ack_n_ > 0) {
                ack_buf_[ack_n_++] = b;
                if (ack_n_ < drive_frame::ACK_LEN) continue;
                ack_n_ = 0;
                drive_frame::Ack a{};
                if (drive_frame::decode_ack(ack_buf_, a)) on_ack(a, now_s);
                else st_.n_ack_bad++;
                continue;
            }
            if (b == '\n' || b == '\r') {
                if (!pending_.empty()) {
                    if (lines) lines->push_back(pending_);
                    pending_.clear();
                }
            } else if (pending_.size() < 256) {
                pending_.push_back((char)b);
            } else {
                pending_.clear();       // overflow guard: garbage, not a line we lost
            }
        }
        if (n < (ssize_t)sizeof(buf)) break;
    }
}

void SerialLink::on_ack(const drive_frame::Ack& a, double now_s) {
    st_.n_ack++;
    st_.fw_n_rx = a.n_rx;
    st_.fw_n_crc = a.n_crc;
    st_.fw_state = a.state;
    const double sent = sent_at_[a.seq_echo % RTT_RING];
    // Host -> board -> host, and it contains BOTH USB quanta. Reported as an upper bound
    // on one-way latency, never halved: the two directions are not symmetric and there is
    // nothing here that measures the split.
    if (sent > 0.0 && now_s >= sent && now_s - sent < 1.0) st_.rtt_s.push_back(now_s - sent);
}

}  // namespace pmw
