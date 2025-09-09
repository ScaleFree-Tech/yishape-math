package com.reremouse.lab.data;

import com.reremouse.lab.math.IVector;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author lteb2
 */
public class Column  implements Serializable{
    
    private String name;//列名
    private ColumnType columnType = ColumnType.Float;//列的类型
    
    private List<Object> data = new ArrayList();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public ColumnType getColumnType() {
        return columnType;
    }

    public void setColumnType(ColumnType columnType) {
        this.columnType = columnType;
    }

    public List<Object> getData() {
        return data;
    }

    public void setData(List<Object> data) {
        this.data = data;
    }
    
    
    
    /**
     * 如果列类型为Float，返回；如果是String，报错
     * @return 
     */
    public IVector toVec() {
        return IVector.of(data.toArray(Float[]::new));
    }
    
    /**
     * 如果列类型为String，返回；如果是Float，转换为String返回
     * @return 
     */
    public List<String> toStringList() {
        return data.stream().map(e->e.toString()).toList();
    }
}
