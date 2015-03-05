/*
 * TimeUtil.h
 *
 *  Created on: Feb 26, 2015
 *      Author: xxy
 */

#ifndef SRC_PREPROCESSING_UTILS_TIMEUTIL_H_
#define SRC_PREPROCESSING_UTILS_TIMEUTIL_H_

#include <stdio.h>
#include <sys/timeb.h>

struct timeb t;
long long getSystemTime() {
    ftime(&t);
    return 1000 * t.time + t.millitm;
}


#endif /* SRC_PREPROCESSING_UTILS_TIMEUTIL_H_ */
