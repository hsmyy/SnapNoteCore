/*
 * configUtil.h
 *
 *  Created on: Feb 3, 2015
 *      Author: xxy
 */

#ifndef IMAGE_PROCESS_SRC_UTIL_CONFIGUTIL_H_
#define IMAGE_PROCESS_SRC_UTIL_CONFIGUTIL_H_

/*
 *parameter: cfgfilepath 文件的绝对路径名如: /user/home/my.cfg
 *           key         文本中的变量名
 *           value       对应变量的值，用于保存
 *
 */
#include <iostream>
#include <string>
#include <fstream>
#include <map>
using namespace std;
class Config {
private:
	vector<pair<string, string> > configs;
	bool contains(string key) {
		for (int i = 0; i < configs.size(); i++) {
			pair<string, string> cur = configs[i];
			if (cur.first == key)
				return true;
		}
		return false;
	}
public:
	Config(string cfgfilepath) {
		readConfigFile(cfgfilepath);
	}
	vector<pair<string, string> > readConfigFile(string cfgfilepath) {
		configs.clear();
		fstream cfgFile;
		cfgFile.open(cfgfilepath.c_str()); //打开文件
		if (!cfgFile.is_open()) {
			cerr << "can not open cfg file!" << endl;
			return configs;
		}
		char tmp[1000];
		while (!cfgFile.eof()) //循环读取每一行
		{
			cfgFile.getline(tmp, 1000); //每行读取前1000个字符，1000个应该足够了
			string line(tmp);
			size_t pos = line.find('='); //找到每行的“=”号位置，之前是key之后是value
			if (pos == string::npos)
				return configs;
			string key = line.substr(0, pos); //取=号之前
			string value = line.substr(pos + 1); //取=号之后
			//if (!contains(key)) {
			configs.push_back(pair<string, string>(key, value));
//			} else {
//				cerr << "config " << key << " : " << value
//						<< " has been override!" << endl;
//			}
		}
		return configs;
	}
	string getAndErase(string key) {
		vector<pair<string, string> >::iterator iter = configs.begin();
		for (; iter != configs.end(); iter++) {
			if (iter->first == key) {
				string value = iter->second;
				iter = configs.erase(iter);
				return value;
			}
		}
		return "";
	}
	string get(string key) {
		vector<pair<string, string> >::iterator iter = configs.begin();
		for (; iter != configs.end(); iter++) {
			if (iter->first == key) {
				return iter->second;
			}
		}
		return "";
	}
	pair<string, string> get(int index) {
		return configs[index];
	}
	int size() {
		return configs.size();
	}

};

#endif /* IMAGE_PROCESS_SRC_UTIL_CONFIGUTIL_H_ */
