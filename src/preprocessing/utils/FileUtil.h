/*
 * utils.h
 *
 *  Created on: Jan 13, 2015
 *      Author: xxy
 */

#ifndef PREPROCESSING_SRC_UTILS_H_
#define PREPROCESSING_SRC_UTILS_H_

#include <iostream>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <fstream>
using namespace std;

class FileUtil {
public:
	static vector<string> getAllFiles(string path) {
		vector<string> files;
		struct dirent *de = NULL;
		DIR *d = NULL;

		d = opendir(path.c_str());
		if (d == NULL) {
			cerr << "Couldn't open directory: "<<path.c_str() << endl;
			return files;
		}

		// Loop while not NULL
		while (de = readdir(d)) {
			if (strcmp(de->d_name, "..") == 0 || strcmp(de->d_name, ".") == 0)
				continue;
			//cout << de->d_name << endl;
			if (de->d_name)
				files.push_back(de->d_name);
		}

		closedir(d);
		return files;
	}

	static string getFileName(string filepath) {
		int start = filepath.find_last_of('/');
		if (start < 0)
			start = -1;

		//cout << filepath.substr(start + 1, end - start - 1) << endl;
		return filepath.substr(start + 1);
	}

	static string getFileNameNoSuffix(string filepath) {
		int start = filepath.find_last_of('/');
		int end = filepath.find_last_of(".");
		if (start < 0)
			start = -1;
		if (end < 0)
			end = string::npos;

		//cout << filepath.substr(start + 1, end - start - 1) << endl;
		return filepath.substr(start + 1, end - start - 1);
	}
	static void rmFile(string filepath) {
		string cmd = string("rm ") + filepath;
		system(cmd.c_str());
	}
	static void writeToFile(const string text, const string filePath) {
		ofstream out(filePath.c_str());
		out << text;
		out.close();
	}

	static void cleanText(string& str, const string& from, const string& to) {
		string reg = "[^a-zA-Z0-9]+";
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != string::npos) {
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}
};

#endif /* PREPROCESSING_SRC_UTILS_H_ */
