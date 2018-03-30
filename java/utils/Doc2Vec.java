package com.demon.utils;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import javax.swing.plaf.synth.SynthSpinnerUI;

import org.omg.CORBA.portable.ValueInputStream;

import com.demon.utils.NLP.Feature;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.mining.word2vec.DocVectorModel;
import com.hankcs.hanlp.mining.word2vec.Vector;
import com.hankcs.hanlp.mining.word2vec.WordVectorModel;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;

public class Doc2Vec {

	public static void segment(String loadFile, String outFileName) {
		List<Feature> sentences = NLP.loadData(loadFile);
		DWriter out = new DWriter(outFileName);
		int cnt = 0;
		for (Feature sent : sentences) {
			List<Term> termList = NLPTokenizer.segment(sent.sentence);
			if (sent.user.equals("b092aa49-4688-3c41-8aec-43c03186567f"))
				System.out.println(termList);
			StringBuilder sb = new StringBuilder();
			for (Term term : termList) {
				if (term.nature == Nature.w || term.nature == Nature.nx)
					continue;
				sb.append(term.word + " ");
			}
			if (sb.length() != 0)
				sb.deleteCharAt(sb.length() - 1);
			out.println(sb.toString());
			System.out.println(cnt++);
		}
		out.close();
	}

	public static void createDataset() {
		segment("data/train_first.csv", "data/tourist.txt");
		segment("data/predict_first.csv", "data/tourist.txt");
	}
	

	public static void main(String[] args) throws IOException, NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		// 加载模型
		DocVectorModel docVectorModel = new DocVectorModel(new WordVectorModel("data/tourist.zh.vec"));
		List<Feature> sentences_train = NLP.loadData("data/train_first.csv");
		List<Feature> sentences_predict = NLP.loadData("data/predict_first.csv");
		sentences_train.addAll(sentences_predict);
		
		DWriter out = new DWriter("data/data_tourist_vector.csv");
		List<String> columns = new ArrayList<>();
		columns.add("Id");
		for (int i = 0; i < 200; ++i) {
			columns.add("Vector_" + i);
		}
		out.println(NLP.toHead(columns.toArray(new String[0]), ","));
		int cnt = 0;
		for (Feature sent : sentences_train) {
			Vector v = docVectorModel.addDocument(cnt, sent.sentence);
			if (v == null) {
				System.out.println(sent.sentence);
		        List<String> values = new ArrayList<>();
		        values.add(sent.user);
		        for (int j = 0; j < 200; ++j) {
		        	values.add(String.valueOf(""));
		        }
		        out.println(NLP.toHead(values.toArray(new String[0]), ","));
			}
			else {
				Class<?> classType = v.getClass();
		        Field field = classType.getDeclaredField("elementArray");
		        field.setAccessible(true);
		        float[] m = (float[]) field.get(v);
		        List<String> values = new ArrayList<>();
		        values.add(sent.user);
		        for (int j = 0; j < m.length; ++j) {
		        	values.add(String.valueOf(m[j]));
		        }
		        out.println(NLP.toHead(values.toArray(new String[0]), ","));
			}
	        System.out.println(cnt ++);
		}
		out.close();
	}

}
