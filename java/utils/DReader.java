package com.demon.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class DReader {
	
	private BufferedReader reader;
	private String line;
	
	public DReader(String fileName){
		try {
			File file = new File(fileName);
			FileReader fileReader = new FileReader(file);
			reader = new BufferedReader(fileReader);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public boolean hasNext(){
		try {
			line = reader.readLine();
			if (line == null){
				reader.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return line != null;
	}
	
	public String next(){
		return line;
	}
	
	
	public static void main(String[] args) {
		DReader reader = new DReader("data/train_first.csv");
		while (reader.hasNext()){
			String line = reader.next();
			System.out.println(line);
		}
	}
	

}
