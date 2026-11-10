#include <cstdlib>
#define HIL_HOST 1
#include "../src/hil/hil_main.cpp"
int main() { setup(); for (;;) loop(); }
