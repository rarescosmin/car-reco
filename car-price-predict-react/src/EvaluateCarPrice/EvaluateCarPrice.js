import React, { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import axios from 'axios'

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
    },
    paper: {
        padding: theme.spacing(2),
        textAlign: 'center'
    }
}));


const EvaluateCarPrice = () => {
    const classes = useStyles();

    const [loading, setLoading] = useState(false);

    const [price, setPrice] = useState(null);

    const [make, setMake] = useState('');
    const [model, setModel] = useState('');
    const [year, setYear] = useState('');
    const [mileage, setMileage] = useState('');
    const [fuelType, setFuelType] = useState('');
    const [engineCapacity, setEngineCapacity] = useState('');
    const [cylinders, setCylinders] = useState('');

    const makeChangeHandler = (event) => {
        setMake(event.target.value);
    }

    const modelChangeHandler = (event) => {
        setModel(event.target.value);
    }

    const yearChangeHandler = (event) => {
        setYear(event.target.value);
    }

    const mileageChangeHandler = (event) => {
        setMileage(event.target.value);
    }

    const fuelTypeChangeHandler = (event) => {
        setFuelType(event.target.value);
    }

    const engineCapacityChangeHandler = (event) => {
        setEngineCapacity(event.target.value);
    }

    const cylindersChangeHandler = (event) => {
        setCylinders(event.target.value);
    }

    const evaluateButtonHandler = async () => {
        const requestBody = {
            'make': make,
            'model': model,
            'year': year,
            'mileage': mileage,
            'fuelType': fuelType,
            'engineCapacity': engineCapacity,
            'cylinders': cylinders
        }
        console.log('request body: ', requestBody);

        setLoading(true);
        const response = await axios.post('http://localhost:5000/evaluate_price', requestBody);
        const retreivedPrice = String(response.data.price);

        const parsedPrice = retreivedPrice.split('.')

        setLoading(false);
        setPrice(parsedPrice[0]);
    }

    const recommendationsButtonHandler = () => {

    }

    return (
        // <Paper className={classes.root} style={{maxWidth: '60vw', marginLeft: 'auto', marginRight: 'auto', marginTop: '3vh'}}>
            <Grid container spacing={3} style={{maxWidth: '90vw', marginLeft: 'auto', marginRight: 'auto', marginTop: '3vh'}}>
                <Grid item xs={12}>
                    <Paper className={classes.paper} elevation={3} style={{fontWeight: 'bold'}}>
                        Please fill the form below
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <div>
                            <TextField 
                                size="small" 
                                value={make}
                                onChange={makeChangeHandler}
                                label="Make" 
                                placeholder="Make" />
                        </div>
                        <div>
                            <TextField 
                                size="small" 
                                value={model} 
                                onChange={modelChangeHandler}
                                label="Model" 
                                placeholder="Model"/>
                        </div>
                        <div>
                            <TextField 
                                size="small" 
                                value={year}
                                onChange={yearChangeHandler} 
                                label="Year"
                                placeholder="Year" />
                        </div>
                        <div>
                            <TextField 
                                size="small" 
                                value={mileage}
                                onChange={mileageChangeHandler} 
                                label="Mileage"
                                placeholder="Mileage" />
                        </div>
                        <div>
                            <TextField 
                                size="small" 
                                value={fuelType}
                                onChange={fuelTypeChangeHandler} 
                                label="Fuel Type"
                                placeholder="Fuel Type" />
                        </div>
                        <div>
                            <TextField 
                                size="small" 
                                value={engineCapacity}
                                onChange={engineCapacityChangeHandler} 
                                label="Engine Capacity"
                                placeholder="Engine Capacity" />
                        </div>
                        <div>
                            <TextField 
                                size="small" 
                                value={cylinders}
                                onChange={cylindersChangeHandler} 
                                label="Cylinders"
                                placeholder="Cylinders" />
                        </div>
                        <div style={{marginTop: '1vh'}}>
                            <Button variant="contained" color="primary" onClick={evaluateButtonHandler}>
                                EVALUATE
                            </Button>
                            
                        </div>
                    </Paper>
                    {   
                        loading ? 
                            <Paper className={classes.paper}>
                                Loading.....
                            </Paper>    
                        :
                            price != null ? 
                                <Paper className={classes.paper}>
                                    <p style={{fontWeight: 'bold'}}>Predicted price: {price} €</p>
                                    <Button variant="contained" color="primary" onClick={recommendationsButtonHandler}>
                                        CHECK SIMILAR CARS
                                    </Button>
                                </Paper>
                            :
                                null

                    }
                    
                </Grid>
            </Grid>
    );
};

export default EvaluateCarPrice
