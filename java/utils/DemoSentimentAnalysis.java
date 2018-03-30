package com.demon.utils;

import java.io.File;
import java.io.IOException;
import java.util.List;

import com.demon.utils.NLP.Feature;
import com.hankcs.hanlp.classification.classifiers.IClassifier;
import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier;

/**
 * 第一个demo,演示文本分类最基本的调用方式
 *
 * @author hankcs
 */
public class DemoSentimentAnalysis {
	/**
	 * 中文情感挖掘语料-ChnSentiCorp 谭松波
	 */
	public static final String TOURIST_FOLDER = "data/旅游景点评论";

	private IClassifier classifier;

	public DemoSentimentAnalysis(String dataset) throws IOException {
		File corpusFolder = new File(dataset);
		if (!corpusFolder.exists() || !corpusFolder.isDirectory()) {
			System.err.println("没有文本分类语料，请阅读IClassifier.train(java.lang.String)中定义的语料格式、准备语料");
			System.exit(1);
		}
		classifier = new NaiveBayesClassifier();
		classifier.train(TOURIST_FOLDER);
	}

	public int predict(String text) {
		String pred = classifier.classify(text);
		if (pred.equals("正面"))
			return 1;
		else
			return 0;
	}
	
	public void write(List<Feature> sentences){
		int negCnt = 0;
		int posCnt = 0;
		for (Feature feature : sentences) {
			if (feature.label.equals("1") || feature.label.equals("2") || feature.label.equals("3")) {
				DWriter out = new DWriter(TOURIST_FOLDER + "/负面/neg." + negCnt + ".txt");
				out.println(feature.sentence);
				out.close();
				negCnt ++;
				System.out.println(negCnt);
			}
			if (feature.label.equals("5")) {
				DWriter out = new DWriter(TOURIST_FOLDER + "/正面/pos." + posCnt + ".txt");
				out.println(feature.sentence);
				out.close();
				posCnt ++;
				System.out.println(posCnt);
			}
		}
		System.out.println("数据库写入完毕");
	}
	
	public static void senti(String dataset, String loadFile, String outFilename, String featureName) throws IOException {
		DemoSentimentAnalysis sentAnalysis = new DemoSentimentAnalysis(dataset);
		List<Feature> sentences = NLP.loadData(loadFile);
		DWriter out = new DWriter(outFilename);
		out.println(NLP.toHead(new String[] { "Id", featureName}, ","));
		int cnt = 0;
		for (Feature sent : sentences) {
			int label = sentAnalysis.predict(sent.sentence);
			out.println(NLP.toHead(new String[] {sent.user, label + ""}, ","));
			System.out.println(cnt++);
		}
		out.close();
	}
	
	// 自身数据库 得到的情感分析
	public static void run(String dataset) throws IOException {
		DemoSentimentAnalysis sentAnalysis = new DemoSentimentAnalysis(dataset);
		List<Feature> sentences = NLP.loadData("./data/train_first.csv");
		DWriter out = new DWriter("./data/train_sentiment_self.csv");
		out.println(NLP.toHead(new String[] { "Id", "sentiment_self" }, ","));
		int cnt = 0;
		for (Feature sent : sentences) {
			int label = sentAnalysis.predict(sent.sentence);
			out.println(NLP.toHead(new String[] {sent.user, label + "" }, ","));
			System.out.println(cnt++);
		}
		out.close();

		sentences = NLP.loadData("./data/predict_first.csv");
		out = new DWriter("./data/predict_sentiment_self.csv");
		out.println(NLP.toHead(new String[] { "Id", "sentiment_self" }, ","));
		cnt = 0;
		for (Feature sent : sentences) {
			int label = sentAnalysis.predict(sent.sentence);
			out.println(NLP.toHead(new String[] { sent.user, label + ""}, ","));
			System.out.println(cnt++);
		}
		out.close();
	}
	
	public static void main(String[] args) throws IOException {
		senti(TOURIST_FOLDER, "data/train_score12.csv", "data/train_score12_self.csv", "sentiment_self");
	}

}
