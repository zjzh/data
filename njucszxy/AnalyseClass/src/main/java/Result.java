import java.util.ArrayList;
import java.util.List;

public class Result {
    //保存至JSON格式的结果
    public String interfaceName = new String();                         //接口名
    public List<String> fatherList = new ArrayList<String>();           //接口的父亲
    public List<String> classRealized = new ArrayList<String>();        //所有实现该接口的类
}
