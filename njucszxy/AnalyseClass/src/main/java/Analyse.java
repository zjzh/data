import org.apache.commons.io.FileUtils;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Analyse {
    //我自己放数据的路径
    public String JavaPath = new String("C:/Users/njucszxy/Desktop/data/java");
    public String SwiftPath = new String("C:/Users/njucszxy/Desktop/data/swift");

    //程序内部存储数据
    private List<Element> classList = new ArrayList<Element>();
    private List<Element> interfaceList = new ArrayList<Element>();
    private List<Result> resultList = new ArrayList<Result>();
    private int[] vector;

    //扫描文件
    public void ScanFiles(String filePath)
    {
        File root = new File(filePath);
        File[] files = root.listFiles();
        for(File file:files)
        {
            if(file.isDirectory())
                ScanFiles(file.getAbsolutePath());          //递归
            else
                AnalyseFile(file.getAbsolutePath());
        }
    }
    //解析JSON文件
    public void AnalyseFile(String filePath)
    {
        File file = new File(filePath);
        try {
            //解析JSON文件并分类存储至程序内部
            String content = FileUtils.readFileToString(file);
            JSONObject jsonObject = new JSONObject(content);
            Element element = new Element();
            element.className = jsonObject.getString("class_name");
            element.classType = jsonObject.getString("class_type");
            JSONArray child = jsonObject.getJSONArray("subclass_list");
            for(int i = 0;i < child.length();i++)
                element.childrenList.add((String) child.get(i));
            JSONArray father = jsonObject.getJSONArray("class_inherit_list");
            for(int i = 0;i < father.length();i++)
                element.fatherList.add((String) father.get(i));
            JSONArray inter = jsonObject.getJSONArray("interface_list");
            for(int i = 0;i < inter.length();i++)
                element.interfaceList.add((String) inter.get(i));
            if(element.classType.equals("class") || element.classType.equals("struct"))
                classList.add(element);
            else if(element.classType.equals("interface") || element.classType.equals("protocol"))
                interfaceList.add(element);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
    //存储接口们的继承关系至结果单元
    public void getFather()
    {
        for(Iterator<Element> i = interfaceList.iterator(); i.hasNext();)
        {
            Result result = new Result();
            Element element = i.next();
            result.interfaceName = element.className;
            result.fatherList = element.fatherList;
            result.classRealized = new ArrayList<String>();
            resultList.add(result);
        }
    }
    //寻找并存储接口们的实现类至结果单元
    public void getClassRealized()
    {
        //测试
        for(int i = 0;i < classList.size();i++)
            System.out.println(classList.get(i).className + " : " + classList.get(i).interfaceList);
        //使用线性表实现的接口继承树
        vector = new int[interfaceList.size()];
        //计算继承关系，将数组每个项填充为其父接口在数组中的下标，若没有父接口则设为-1
        for(int i = 0;i < interfaceList.size();i++)
        {
            if(interfaceList.get(i).fatherList.size() == 0)
            {
                vector[i] = -1;
                continue;
            }
            String fatherName = interfaceList.get(i).fatherList.get(0);
            boolean flag = false;
            for(int j = 0;j < interfaceList.size();j++)
            {
                if(interfaceList.get(j).className.equals(fatherName))
                {
                    flag = true;
                    vector[i] = j;
                    break;
                }
            }
            if(!flag)
                vector[i] = -1;
        }
        //在所有类中寻找每个接口的实现类
        for(int i = 0;i < interfaceList.size();i++)
        {
            String interfaceName = interfaceList.get(i).className;
            System.out.println(interfaceName);
            for(int j = 0;j < classList.size();j++)
            {
                String className = classList.get(j).className;
                if(classList.get(j).interfaceList.contains(interfaceName))
                {
                    resultList.get(i).classRealized.add(className);
                    System.out.println("add:" + className);
                }
            }
            if(vector[i] != -1)
                System.out.println("Call:" + interfaceName +"," + interfaceList.get(vector[i]).className);
            else
                System.out.println("Call:" + interfaceName +",null");
            //通知自己的父接口更新实现类信息
            callUpdate(i,vector[i]);
        }
    }
    //递归调用，直到通知到根更新实现类信息
    private void callUpdate(int callerID,int fatherID)
    {
        //根
        if(fatherID == -1)
            return;
        //将子接口实现类与自身的实现类取交集
        for(int i = 0;i < resultList.get(callerID).classRealized.size();i++)
        {
            String className = resultList.get(callerID).classRealized.get(i);
            if(!resultList.get(fatherID).classRealized.contains(className))
            {
                resultList.get(fatherID).classRealized.add(className);
                System.out.println("review:"+ className);
            }
        }
        //递归调用
        callUpdate(fatherID,vector[fatherID]);
    }
    //存储结果单元至给定本地路径
    public void saveFiles(String filePath)
    {
        try {
            for(int i = 0;i < resultList.size();i++)
            {
                String resultFilePath = filePath + "/" + resultList.get(i).interfaceName + ".json";
                BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(resultFilePath));
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("name",resultList.get(i).interfaceName);
                jsonObject.put("father",resultList.get(i).fatherList);
                jsonObject.put("class",resultList.get(i).classRealized);
                String content = jsonObject.toString();
                bufferedWriter.write(content);
                bufferedWriter.flush();
                bufferedWriter.close();
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
    public static void main(String args[])
    {
        //JAVA
        Analyse analyse = new Analyse();
        analyse.ScanFiles(analyse.JavaPath);
        analyse.getFather();
        analyse.getClassRealized();
        analyse.saveFiles("C:/Users/njucszxy/Desktop/result/java");
        //Swift
        analyse = new Analyse();
        analyse.ScanFiles(analyse.SwiftPath);
        analyse.getFather();
        analyse.getClassRealized();
        analyse.saveFiles("C:/Users/njucszxy/Desktop/result/swift");
    }
}
