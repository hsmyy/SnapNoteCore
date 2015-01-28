/*
 * util.h
 *
 *  Created on: Jan 28, 2015
 *      Author: fc
 */

#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#include <dirent.h>
#include <iostream>

using namespace std;

vector<string> listFiles(string folder) {
	DIR *dir;
	struct dirent *ent;
	vector<string> files;
	if ((dir = opendir(folder.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			if (strlen(ent->d_name) > 0 && ent->d_name[0] != '.') {
				files.push_back(ent->d_name);
			}
		}
	}
	return files;
}

void saveToFile(string filePath, string text){
	ofstream fileStream;
	fileStream.open(filePath.c_str());
	fileStream << text;
	fileStream.close();
}


#endif /* UTIL_UTIL_H_ */
