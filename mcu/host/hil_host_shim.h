/* Minimal Arduino stand-in to run the HIL firmware on the host over a pty,
 * for testing the protocol and mcu/hil.py without a board. */
#pragma once
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#define PROGMEM
static inline uint32_t micros() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint32_t)(t.tv_sec * 1000000ull + t.tv_nsec / 1000); }
struct HostSerial {
  int fd = -1;
  void begin(long) { fd = atoi(getenv("HIL_FD")); }
  int read() { uint8_t c; return ::read(fd, &c, 1) == 1 ? c : -1; }
  void w(const char *s) { ::write(fd, s, strlen(s)); }
  void print(const char *s) { w(s); }
  void print(long v) { char b[32]; snprintf(b, 32, "%ld", v); w(b); }
  void print(unsigned long v) { char b[32]; snprintf(b, 32, "%lu", v); w(b); }
  void print(int v) { print((long)v); }
  void print(unsigned v) { print((unsigned long)v); }
  void print(double v, int dig) { char b[64]; snprintf(b, 64, "%.*f", dig, v); w(b); }
  void print(char c) { char b[2] = {c, 0}; w(b); }
  void println() { w("\r\n"); }
  template <class T> void println(T v) { print(v); println(); }
} Serial;
