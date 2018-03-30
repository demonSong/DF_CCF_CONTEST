package com.demon.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class DWriter {
	
	private FileWriter out;
	private File file;
	
	public DWriter(String fileName){
		file = new File(fileName);
		if (!file.exists()){
			try {
				file.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		try {
			out = new FileWriter(file, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void print(String str){
		try {
			out.write(str);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void println(String str){
		try {
			out.write(str + "\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void close(){
		try {
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public static void main(String[] args) {
		DWriter out = new DWriter("data/test.txt");
		out.println("我是测试文件");
		out.close();
	}

}
