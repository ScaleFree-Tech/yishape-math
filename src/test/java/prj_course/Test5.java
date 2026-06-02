package prj_course;

import com.yishape.lab.math.stats.Stats;

import model_zoo.naive_news_vendor.NewsvendorModel;

public class Test5 {
    public static void main(String[] args){
        double cost = 100;
        double price = 150;
        double loss = 50;

        //分布假设：正态
        double mean = 120;
        double var = (0.1*Math.pow((80-mean),2)
                    +0.2*Math.pow((1000-mean),2)
                    +0.4*Math.pow((120-mean),2)
                    +0.2*Math.pow((140-mean),2)
                    +0.1*Math.pow((160-mean),2))/5.0;
        double std = Math.sqrt(var);
        System.out.println("mean:"+mean+", std:"+std);
        //分布函数
        var dist = Stats.norm(mean, std);
        var model = new NewsvendorModel(cost,price,loss,dist);
        var quantity = model.computeTheoreticalOptimalQuantity();
        var profit = model.computeExpectedProfit(quantity);
        System.out.println(quantity+":"+profit);
    }
}
