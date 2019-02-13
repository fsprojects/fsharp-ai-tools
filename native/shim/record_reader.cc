

#include "record_reader.h"
#include "utilities.h"

//#include <string.h>

//#include "tensorflow/c/record_reader.h"
//#include "tensorflow/c/tf_status_helper.h"
#include "../include/tensorflow/core/lib/io/record_reader.h"

int TwoTimes(int x) { return 2 * x; }
void TwoTimes2(int x, Callback call) {
	call(x, 2 * x);
}
