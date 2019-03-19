import java.util.ArrayList;
import java.util.List;

public class Element {
    //存储读取的JSON数据的单元
    public String className = new String();
    public String classType = new String();
    public List<String> childrenList = new ArrayList<String>();
    public List<String> fatherList = new ArrayList<String>();
    public List<String> interfaceList = new ArrayList<String>();
    Element()
    {
        ;
    }
    Element(Element e)
    {
        className = e.className;
        classType = e.classType;
        childrenList = e.childrenList;
        fatherList = e.fatherList;
        interfaceList = e.interfaceList;
    }
}
