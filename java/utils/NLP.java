package com.demon.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;

public class NLP {

	static class Feature {
		String user;
		String sentence;
		String label;

		Feature(String user, String sentence, String label) {
			this.user = user;
			this.sentence = sentence;
			this.label = label;
		}

		Feature(String user, String sentence) {
			this.user = user;
			this.sentence = sentence;
		}
	}

	public static List<Feature> loadData(String filename) {
		List<Feature> sentences = new ArrayList<>();
		DReader reader = new DReader(filename);
		int cnt = 0;
		while (reader.hasNext()) {
			String[] line = reader.next().split(",");
			if (cnt != 0)
				sentences.add(new Feature(line[0], line[1]));
			cnt++;
		}
		System.out.println(sentences.size());
		return sentences;
	}

	public static String toHead(String[] columns, String sep) {
		return String.join(sep, columns);
	}

	public static void segment(String loadFile, String outFileName) {
		List<Feature> sentences = loadData(loadFile);
		DWriter out = new DWriter(outFileName);
		out.println(toHead(new String[] { "Id", "words" }, ","));
		int cnt = 0;
		for (Feature sent : sentences) {
			List<Term> termList = NLPTokenizer.segment(sent.sentence);

			StringBuilder sb = new StringBuilder();
			for (Term term : termList) {
				if (term.nature == Nature.w) continue;
				sb.append(term.word + ";");
			}
			if (sb.length() != 0) sb.deleteCharAt(sb.length() - 1);
			out.println(toHead(new String[] { sent.user, "[" + sb.toString() + "]" }, ","));
			System.out.println(cnt++);
		}
		out.close();
	}
	
	public static void detectAddress(String loadFile, String outFile) {
		Segment segment = HanLP.newSegment().enablePlaceRecognize(true);
		// 加载数据集
		List<Feature> sentences = loadData(loadFile);
		DWriter out = new DWriter(outFile);
		out.println(toHead(new String[] { "Id", "address_list" }, ","));
		int cnt = 0;
		for (Feature f : sentences) {
			List<Term> termList = segment.seg(f.sentence);
			StringBuilder sb = new StringBuilder();
			for (Term term : termList) {
				if (term.nature == Nature.ns) {
					sb.append(term.word + ";");
				}
			}
			if (sb.length() != 0) sb.deleteCharAt(sb.length() - 1);
			out.println(toHead(new String[] { f.user, "[" + sb.toString() + "]" }, ","));
			System.out.println(cnt++);
		}
		out.close();
	}
	
	public static Map<String, Integer> initPropertyMap(){
		Map<String, Integer> map = new HashMap<>();
		for (Nature n : Nature.values()) {
			map.put(n.toString(), map.getOrDefault(n.toString(), 0));
		}
		return map;
	}
	
	public static void totalProperty(String loadFile, String outFile) { 
		List<Feature> sentences = loadData(loadFile);
		int cnt = 0;
		DWriter out = new DWriter(outFile);
		List<String> headers = new ArrayList<>();
		headers.add("Id");
		for (Nature n : Nature.values()) headers.add(n.toString());
		for (Nature n : Nature.values()) headers.add(n.toString() + "_ratio");
		headers.add("entropy");
		out.println(toHead(headers.toArray(new String[0]), ","));
		for (Feature f : sentences) {
			Map<String, Integer> map = initPropertyMap();
			List<Term> termList = NLPTokenizer.segment(f.sentence);
			for (Term t : termList) {
				String property = t.nature.toString();
				map.put(property, map.get(property) + 1);
			}
			List<String> line = new ArrayList<>();
			line.add(f.user);
			for (Nature n : Nature.values()) line.add(map.get(n.toString()) + "");
			double entropy_sum = 0;
			for (Nature n : Nature.values()) {
				double count = map.get(n.toString());
				count /= termList.size();
				line.add(String.valueOf(count));
				if (count > 0) {
					double entropy = -count * Math.log(count);
					entropy_sum += entropy;
				}
			}
			line.add(String.valueOf(entropy_sum));
			out.println(toHead(line.toArray(new String[0]), ","));
			cnt += 1;
			System.out.println(cnt);
		}
		out.close();
		
	}

	public static void main(String[] args) {
		totalProperty("data/train_first.csv", "data/train_property.csv");
		totalProperty("data/predict_first.csv", "data/predict_property.csv");
	}
}
