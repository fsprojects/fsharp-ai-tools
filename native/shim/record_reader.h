#include "utilities.h"

// NOTE stub functions in for the moment
typedef int(__stdcall* Callback)(
	int x,
	int y);

extern "C" {
	DLLEXTERN int TwoTimes(int x);
	DLLEXTERN void TwoTimes2(int x, Callback call);
}
